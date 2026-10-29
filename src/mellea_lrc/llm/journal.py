"""Reuse durably recorded reviews when an interrupted substage resumes.

The journal belongs to a run, beside its native Document. It does not change
Document or IvrRun serialization. A run owner must validate its source and
execution configuration before opening the journal.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections import Counter, defaultdict, deque
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import TYPE_CHECKING

from pydantic import BaseModel

from mellea_lrc.llm.config import PROFILE_OPTION
from mellea_lrc.model.ivr import IvrRun

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator
    from types import TracebackType

    from mellea import MelleaSession
    from mellea.stdlib.sampling import MultiTurnStrategy

    from mellea_lrc.llm.ivr import InstructIvrSpec
    from mellea_lrc.model.document import Document


_ACTIVE: ContextVar[ReviewJournal | None] = ContextVar("mellea_lrc_review_journal", default=None)


def active_journal() -> ReviewJournal | None:
    return _ACTIVE.get()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _ivrs(value: object) -> Iterator[dict]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "ivr" and isinstance(child, dict) and "attempts" in child:
                yield child
            else:
                yield from _ivrs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _ivrs(child)


def _validator(requirement) -> dict[str, str | None]:
    function = requirement.validation_fn
    if function is None:
        return {"name": None, "implementation_sha256": None}
    try:
        source = inspect.getsourcefile(function)
    except TypeError:
        source = None

    def captured_json(value):
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        raise TypeError("Review validator captures unsupported state; use JSON, dataclass or Pydantic values")

    state = {
        "defaults": getattr(function, "__defaults__", None),
        "keyword_defaults": getattr(function, "__kwdefaults__", None),
        "closure": {
            name: cell.cell_contents
            for name, cell in zip(function.__code__.co_freevars, function.__closure__ or (), strict=True)
        }
        if inspect.isfunction(function)
        else {},
        "bound_instance": getattr(function, "__self__", None)
        if inspect.ismethod(function)
        else function
        if not inspect.isfunction(function)
        else None,
    }
    encoded = json.dumps(
        state,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=captured_json,
    )
    return {
        "name": f"{function.__module__}.{function.__qualname__}",
        "implementation_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if source is not None and Path(source).is_file()
        else None,
        "captured_state_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def _specification(session, spec, strategy, model_options) -> dict:
    # Use the same JSON projection as the native trace. The wrapper adds the
    # repair budget to profile metadata, so normalize it before matching too.
    from mellea_lrc.llm.ivr import _run_metadata

    options = dict(model_options)
    if PROFILE_OPTION in options:
        options[PROFILE_OPTION] = {**options[PROFILE_OPTION], "max_attempts": strategy.loop_budget}
    return {
        **_run_metadata(session, spec, options),
        "backend_module": type(session.backend).__module__,
        "strategy": f"{type(strategy).__module__}.{type(strategy).__qualname__}",
        "max_attempts": strategy.loop_budget,
        "requirements": [
            {"description": requirement.description, "validator": _validator(requirement)}
            for requirement in spec.requirements
        ],
    }


class ReviewJournal:
    """One run's completed IVRs, matched by their complete review specification.

    Entries already present in the checkpoint Document are excluded. Remaining
    entries are consumed once in their original call order, including failures.
    New completed reviews are saved immediately, before their substage returns.
    An interrupted call without a durable result can require another request.
    """

    def __init__(
        self,
        directory: Path,
        document: Document | None = None,
        *,
        before_review: Callable[[], None] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._before_review = before_review
        self._lock = Lock()
        self._token = None
        self._pending: dict[str, deque[tuple[dict, IvrRun]]] = defaultdict(deque)
        self._next_id = 1
        self.reused = 0
        self.recorded = 0
        committed = (
            Counter(_digest(value) for value in _ivrs(document.model_dump(mode="json")))
            if document is not None
            else Counter()
        )
        for path in sorted(self.directory.glob("*.json")):
            if not path.stem.isdigit():
                raise ValueError(f"Unexpected review journal entry: {path}")
            self._next_id = max(self._next_id, int(path.stem) + 1)
            entry = json.loads(path.read_text(encoding="utf-8"))
            if set(entry) != {"specification", "signature", "ivr", "ivr_sha256"}:
                raise ValueError(f"Invalid review journal entry: {path}")
            if _digest(entry["specification"]) != entry["signature"]:
                raise ValueError(f"Review journal specification changed: {path}")
            if _digest(entry["ivr"]) != entry["ivr_sha256"]:
                raise ValueError(f"Review journal result changed: {path}")
            run = IvrRun.model_validate(entry["ivr"])
            self._validate_result(entry["specification"], run)
            if committed[entry["ivr_sha256"]]:
                committed[entry["ivr_sha256"]] -= 1
            else:
                self._pending[entry["signature"]].append((entry["specification"], run))

    def __enter__(self) -> ReviewJournal:
        if self._token is not None or active_journal() is not None:
            raise RuntimeError("A review journal is already active in this context")
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._token is None:
            raise RuntimeError("The review journal context is not active")
        _ACTIVE.reset(self._token)
        self._token = None

    @staticmethod
    def _validate_result(specification: dict, run: IvrRun) -> None:
        values = run.model_dump(mode="json")
        for name in (
            "backend",
            "model",
            "model_options",
            "instruction",
            "prefix",
            "grounding_context",
            "user_variables",
            "output_schema",
        ):
            if values[name] != specification[name]:
                raise ValueError(f"Review journal result does not match its specification: {name}")
        if len(run.attempts) > specification["max_attempts"]:
            raise ValueError("Review journal result exceeds its repair budget")

    async def run(
        self,
        session: MelleaSession,
        spec: InstructIvrSpec,
        *,
        strategy: MultiTurnStrategy,
        model_options: dict[str, object],
        invoke: Callable[..., Awaitable[IvrRun]],
    ) -> IvrRun:
        if self._before_review is not None:
            self._before_review()
        specification = _specification(session, spec, strategy, model_options)
        signature = _digest(specification)
        with self._lock:
            if self._pending[signature]:
                saved_specification, run = self._pending[signature].popleft()
                if saved_specification != specification:
                    raise ValueError("Review journal signature collision")
                self.reused += 1
                return run
            identifier = self._next_id
            self._next_id += 1
        run = await invoke(session, spec, strategy=strategy, model_options=model_options)
        self._validate_result(specification, run)
        entry = {
            "specification": specification,
            "signature": signature,
            "ivr": run.model_dump(mode="json"),
            "ivr_sha256": _digest(run.model_dump(mode="json")),
        }
        destination = self.directory / f"{identifier:06d}.json"
        temporary = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                prefix=f".{destination.name}-",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(entry, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            if destination.exists():
                raise FileExistsError(f"Review journal entry already exists: {destination}")
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        with self._lock:
            self.recorded += 1
        return run
