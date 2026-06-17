"""Modal server for the reusable CourtListener access API."""

from pathlib import Path

import modal

from mellea_lrc.courtlistener.api import create_api

APP_NAME = "courtlistener-access"
REMOTE_ROOT = Path("/root")
REMOTE_PACKAGE_DIR = REMOTE_ROOT / "mellea_lrc"
LOCAL_REPO_PARENT_INDEX = 3


def _package_dir() -> Path:
    path = Path(__file__).resolve()
    if len(path.parents) > LOCAL_REPO_PARENT_INDEX:
        return path.parents[LOCAL_REPO_PARENT_INDEX] / "src" / "mellea_lrc"
    return REMOTE_PACKAGE_DIR


PACKAGE_DIR = _package_dir()

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "boto3>=1.34",
        "fastapi>=0.115",
        "python-multipart>=0.0.9",
        "requests>=2.32",
    )
    .add_local_dir(PACKAGE_DIR, REMOTE_PACKAGE_DIR, copy=True)
    .env({"PYTHONPATH": str(REMOTE_ROOT)})
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
