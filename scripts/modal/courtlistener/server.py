"""Modal server for the reusable CourtListener access API."""

import os
from random import SystemRandom

import modal

from mellea_lrc.courtlistener.api import create_api

APP_NAME = "courtlistener-access"
TOKEN_ENV_PREFIX = "COURTLISTENER_API_TOKEN"  # noqa: S105
MIN_TOKEN_SHUFFLE_COUNT = 2
TOKEN_SHUFFLER = SystemRandom()

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "boto3>=1.34",
        "fastapi>=0.115",
        "python-multipart>=0.0.9",
        "requests>=2.32",
    )
    .add_local_python_source("mellea_lrc")
)


def _shuffle_courtlistener_tokens() -> None:
    token_keys = sorted(
        key for key, value in os.environ.items() if key.startswith(TOKEN_ENV_PREFIX) and value.strip()
    )
    if len(token_keys) < MIN_TOKEN_SHUFFLE_COUNT:
        return

    token_values = [os.environ[key] for key in token_keys]
    TOKEN_SHUFFLER.shuffle(token_values)
    for key, value in zip(token_keys, token_values, strict=True):
        os.environ[key] = value


@app.function(
    image=image,
    enable_memory_snapshot=True,
    secrets=[
        modal.Secret.from_name("courtlistener", required_keys=["COURTLISTENER_API_TOKEN_1"]),
        modal.Secret.from_name("courtlistener-r2-cache"),
    ],
    max_containers=1,
    timeout=120,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def web() -> object:
    """Expose the CourtListener access API through Modal."""
    _shuffle_courtlistener_tokens()
    return create_api()
