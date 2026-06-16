"""Modal deployment wrapper for the reusable CourtListener access API."""

from pathlib import Path

import modal

from mellea_lrc.courtlistener.api import create_api

APP_NAME = "cl-access-modal"
PACKAGE_DIR = Path(__file__).resolve().parents[2] / "src" / "mellea_lrc"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "boto3>=1.34",
        "fastapi>=0.115",
        "python-multipart>=0.0.9",
        "requests>=2.32",
    )
    .add_local_dir(PACKAGE_DIR, "/root/mellea_lrc", copy=True)
)

courtlistener_secret = modal.Secret.from_name(
    "courtlistener",
    required_keys=["COURTLISTENER_API_TOKEN_1"],
)
r2_secret = modal.Secret.from_name("courtlistener-r2-cache")


@app.function(
    image=image,
    enable_memory_snapshot=True,
    secrets=[courtlistener_secret, r2_secret],
    max_containers=1,
    timeout=120,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def web() -> object:
    """Expose the CourtListener access API through Modal."""
    return create_api()
