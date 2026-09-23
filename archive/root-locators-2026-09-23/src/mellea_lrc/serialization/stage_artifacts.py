"""Persistence for explicitly decorated ``Document -> Document`` stages.

``@serialize()`` has exactly one job: write the :class:`Document` returned by
the decorated stage.  The runner establishes the artifact directory once with
``artifact_directory(path)``; stages never receive persistence configuration.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

if TYPE_CHECKING:
    from mellea_lrc.model.document import Document


P = ParamSpec("P")
StageT = TypeVar("StageT", bound=Callable[..., Any])
_ARTIFACT_DIRECTORY: ContextVar[Path | None] = ContextVar("mellea_lrc_artifact_directory", default=None)


@contextmanager
def artifact_directory(directory: Path | str) -> Iterator[Path]:
    """Select where ``@serialize()`` writes checkpoints for one run."""
    root = Path(directory)
    token = _ARTIFACT_DIRECTORY.set(root)
    try:
        yield root
    finally:
        _ARTIFACT_DIRECTORY.reset(token)


def serialize() -> Callable[[StageT], StageT]:
    """Persist the successful output of one public ``Document -> Document`` stage.

    The output is written to ``<artifact_directory>/<stage>/<document>.json``. The
    stage name is the final recorded pass, which makes a convenience operation
    such as ``grow_roots`` save under ``root_formation``; a function that has
    not recorded a pass falls back to its Python name.  Artifacts are atomic,
    human-readable JSON and are round-tripped before replacing the old one.

    A runner calls ``artifact_directory(path)`` once around its pipeline. This
    keeps the persistence concern out of every decorated stage call.
    """

    def decorate(stage: StageT) -> StageT:
        if inspect.iscoroutinefunction(stage):

            @functools.wraps(stage)
            async def async_stage(*args: P.args, **kwargs: P.kwargs) -> Document:
                before, stage_args, stage_kwargs = _stage_input(args, kwargs)
                result = await stage(*stage_args, **stage_kwargs)
                before.assert_cumulative_successor(result)
                _persist(stage.__name__, result)
                return result

            return cast(StageT, async_stage)

        @functools.wraps(stage)
        def sync_stage(*args: P.args, **kwargs: P.kwargs) -> Document:
            before, stage_args, stage_kwargs = _stage_input(args, kwargs)
            result = stage(*stage_args, **stage_kwargs)
            before.assert_cumulative_successor(result)
            _persist(stage.__name__, result)
            return result

        return cast(StageT, sync_stage)

    return decorate


def _stage_input(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Document, tuple[Any, ...], dict[str, Any]]:
    """Preserve the caller's checkpoint and an untouched transition baseline."""
    from mellea_lrc.model.document import Document

    if args and isinstance(args[0], Document):
        before = args[0].snapshot()
        return before, (before.snapshot(), *args[1:]), kwargs
    if isinstance(kwargs.get("document"), Document):
        before = kwargs["document"].snapshot()
        return before, args, {**kwargs, "document": before.snapshot()}
    raise TypeError("@serialize() requires a Document as the first argument or 'document' keyword")


def _persist(fallback_stage_name: str, document: Document) -> Path:
    """Round-trip and atomically write one successful stage result."""
    from mellea_lrc.model.document import Document

    root = _ARTIFACT_DIRECTORY.get()
    if root is None:
        raise RuntimeError("@serialize() requires an active artifact_directory(path) context")
    if not isinstance(document, Document):
        msg = f"@serialize() expected {fallback_stage_name} to return Document, got {type(document).__name__}"
        raise TypeError(msg)

    payload = document.model_dump(mode="json")
    # A persistence checkpoint is useful only if the next stage can resume
    # from it. Validate that invariant before replacing a previous artifact.
    restored = Document.model_validate(payload)
    if restored != document:
        raise ValueError(f"{fallback_stage_name} produced a lossy document checkpoint")
    stage_name = document.passes[-1] if document.passes else fallback_stage_name
    path = root / _path_component(stage_name) / f"{_document_name(document)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _document_name(document: Document) -> str:
    """Give a source-backed document a readable, collision-resistant filename."""
    if document.source_path:
        readable = _path_component(Path(document.source_path).stem)
        if readable:
            return readable
    digest = hashlib.sha256(document.text.encode("utf-8")).hexdigest()[:12]
    return f"inline-{digest}"


def _path_component(value: str) -> str:
    """Keep caller-controlled source and pass names inside the artifact root."""
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return result or "document"
