# ruff: noqa: INP001
"""Minimal sync and async Mellea client for an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import TYPE_CHECKING, cast

import httpx
import openai
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend
from mellea.core.base import ModelOutputThunk
from mellea.helpers import get_current_event_loop

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_QUESTION = (
    "What kind of language model are you? If you do not know the exact deployed model name, say so."
)


class MelleaLiteLLMBackend(OpenAIBackend):
    """Configure separate sync and async OpenAI clients with one TLS policy."""

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str,
        api_key: str,
        verify_certificates: bool,
    ) -> None:
        self._verify_certificates = verify_certificates
        super().__init__(
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(verify=verify_certificates),
        )

    @property
    def _async_client(self) -> openai.AsyncOpenAI:
        """Return an event-loop-local client with the same TLS policy."""
        key = id(get_current_event_loop())
        client = self._client_cache.get(key)
        if client is None:
            async_kwargs = {
                name: value for name, value in self._openai_client_kwargs.items() if name != "http_client"
            }
            client = openai.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                http_client=httpx.AsyncClient(verify=self._verify_certificates),
                **async_kwargs,
            )
            self._client_cache.put(key, client)
        return client


def create_session() -> MelleaSession:
    """Create a Mellea session entirely from environment variables."""
    backend = MelleaLiteLLMBackend(
        model_id=_required_env("MELLEA_MODEL_ID"),
        base_url=_required_env("MELLEA_API_BASE"),
        api_key=_required_env("MELLEA_API_KEY"),
        verify_certificates=_bool_env("MELLEA_VERIFY_CERTIFICATES", default=True),
    )
    return MelleaSession(backend)


def ask_sync(question: str) -> str:
    """Ask one question through Mellea's synchronous API."""
    session = create_session()
    output = cast(
        ModelOutputThunk[str],
        session.instruct(question, strategy=None),
    )
    return output.value or ""


async def ask_async(question: str) -> str:
    """Ask one question through Mellea's asynchronous API."""
    session = create_session()
    output = cast(
        ModelOutputThunk[str],
        await session.ainstruct(question, strategy=None),
    )
    return await output.avalue()


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected demonstration mode."""
    args = _parse_args(argv)
    if args.mode in {"sync", "both"}:
        print("[sync]")  # noqa: T201
        print(ask_sync(args.question))  # noqa: T201
    if args.mode in {"async", "both"}:
        print("[async]")  # noqa: T201
        print(asyncio.run(ask_async(args.question)))  # noqa: T201


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--mode", choices=("sync", "async", "both"), default="both")
    return parser.parse_args(argv)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        msg = f"Missing required environment variable: {name}"
        raise RuntimeError(msg)
    return value


def _bool_env(name: str, *, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    msg = f"{name} must be true or false"
    raise RuntimeError(msg)


if __name__ == "__main__":
    main()
