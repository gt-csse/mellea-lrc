"""OpenAI-compatible Mellea backend with explicit TLS client configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import openai
from mellea.backends.openai import OpenAIBackend
from mellea.helpers import get_current_event_loop

if TYPE_CHECKING:
    from mellea.core.base import AbstractMelleaTool, ModelOutputThunk


class MelleaLRCOpenAIBackend(OpenAIBackend):
    """Project-owned extension point for Mellea's OpenAI-compatible backend.

    Mellea forwards the same OpenAI client keyword arguments to ``OpenAI`` and
    ``AsyncOpenAI``. The SDK requires different ``httpx`` client types, so this
    adapter constructs each explicitly with a shared certificate policy.
    """

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str,
        api_key: str,
        certificate_verification: bool,
        model_options: dict[str, object] | None = None,
    ) -> None:
        self._certificate_verification = certificate_verification
        super().__init__(
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            model_options=model_options,
            http_client=httpx.Client(verify=certificate_verification),
        )

    @property
    def certificate_verification(self) -> bool:
        """Return whether the backend verifies server certificates."""
        return self._certificate_verification

    async def post_processing(
        self,
        mot: ModelOutputThunk,
        tools: dict[str, AbstractMelleaTool],
        conversation: list[dict[str, object]],
        thinking: object,
        seed: object,
        _format: object,
    ) -> None:
        """Avoid masking an API exception with missing response metadata."""
        if "oai_chat_response" not in mot._meta:  # noqa: SLF001
            span = mot._meta.pop("_telemetry_span", None)  # noqa: SLF001
            if span is not None:
                from mellea.telemetry import end_backend_span  # noqa: PLC0415

                end_backend_span(span)
            return
        await super().post_processing(
            mot,
            tools=tools,
            conversation=conversation,
            thinking=thinking,
            seed=seed,
            _format=_format,
        )

    @property
    def _async_client(self) -> openai.AsyncOpenAI:
        """Return an event-loop-local async client using the configured TLS policy."""
        key = id(get_current_event_loop())
        client = self._client_cache.get(key)
        if client is None:
            async_kwargs = {
                name: value for name, value in self._openai_client_kwargs.items() if name != "http_client"
            }
            client = openai.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                http_client=httpx.AsyncClient(verify=self._certificate_verification),
                **async_kwargs,
            )
            self._client_cache.put(key, client)
        return client
