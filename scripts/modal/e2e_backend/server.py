"""Modal server for the mellea-lrc end-to-end backend."""

import logging

import modal

from scripts.e2e_backend.api import create_app
from scripts.e2e_backend.label_studio_bridge import LabelStudioBridge, LabelStudioConfig
from scripts.e2e_backend.pipeline import E2EBackend

APP_NAME = "mellea-lrc-prototype"

logger = logging.getLogger(APP_NAME)
logging.basicConfig(level=logging.INFO)

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "docling>=2.0",
        "eyecite>=2.6",
        "fastapi[standard]>=0.115",
        "mellea>=0.3.2",
        "requests>=2.32",
    )
    .add_local_python_source("mellea_lrc", "scripts")
)

backend = E2EBackend()


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("label-studio"),
        modal.Secret.from_name("cl-access-modal"),
        modal.Secret.from_name("mellea-assessment"),
    ],
    gpu="L4",
    memory=4096,
    timeout=1800,
    scaledown_window=120,
    enable_memory_snapshot=True,
)
@modal.asgi_app()
def web() -> object:
    """Expose the E2E backend through FastAPI."""
    web_app = create_app(backend)

    @web_app.post("/predict")
    async def predict(payload: dict[str, object]) -> dict[str, list[object]]:
        bridge = LabelStudioBridge(
            backend,
            LabelStudioConfig.from_env(),
            logger=logger,
        )
        return bridge.predict_tasks(payload.get("tasks", []))

    return web_app
