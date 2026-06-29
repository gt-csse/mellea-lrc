"""Local FastAPI entrypoint for frontend E2E testing."""

from pathlib import Path

from mellea_lrc.core.env import load_env_file
from scripts.e2e_backend.api import create_app

load_env_file(Path(".env"))

app = create_app()
