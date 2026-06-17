"""Label Studio request bridging for the Modal E2E backend."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    import logging

    from scripts.modal.e2e_backend.pipeline import E2EBackend

HTTP_TOKEN_TIMEOUT_SECONDS = 30
HTTP_ASSET_TIMEOUT_SECONDS = 120
HTTP_PATCH_TIMEOUT_SECONDS = 60


@dataclass(frozen=True, slots=True)
class LabelStudioConfig:
    """Runtime configuration for Label Studio task bridging."""

    url: str
    refresh_token: str

    @classmethod
    def from_env(cls) -> LabelStudioConfig:
        """Load Label Studio bridge configuration from Modal secrets."""
        return cls(
            url=os.environ["LS_URL"].rstrip("/"),
            refresh_token=os.environ["LS_ACCOUNT_AUTH"],
        )


class LabelStudioBridge:
    """Translate Label Studio task payloads into E2E backend predictions."""

    def __init__(
        self,
        backend: E2EBackend,
        config: LabelStudioConfig,
        *,
        logger: logging.Logger,
    ) -> None:
        self.backend = backend
        self.config = config
        self.logger = logger

    def predict_tasks(self, tasks: object) -> dict[str, list[object]]:
        """Return Label Studio ML-backend predictions for a task payload."""
        if not isinstance(tasks, list):
            return {"results": []}
        if not tasks:
            return {"results": []}

        access_token = _get_access_token(self.config)
        results = []
        for task in tasks:
            if not isinstance(task, dict):
                results.append({"result": []})
                continue
            try:
                results.append(self._predict_task(task, access_token))
            except Exception:
                self.logger.exception("Failed to process Label Studio task %s", task.get("id"))
                results.append({"result": []})
        return {"results": results}

    def _predict_task(self, task: dict[str, object], access_token: str) -> object:
        asset_path = _extract_pdf_path(task)
        content = _fetch_ls_asset(self.config.url, asset_path, access_token)
        output = self.backend.predict_pdf_bytes(content, _filename_from_path(asset_path))

        task_id = task.get("id")
        data = task.get("data") if isinstance(task.get("data"), dict) else {}
        if task_id is not None:
            _set_task_text(
                self.config.url,
                task_id,
                data,
                asset_path,
                str(output["text"]),
                access_token,
            )
        return output["prediction"]


def _get_access_token(config: LabelStudioConfig) -> str:
    import requests  # noqa: PLC0415

    response = requests.post(
        f"{config.url}/api/token/refresh",
        json={"refresh": config.refresh_token},
        timeout=HTTP_TOKEN_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    access = response.json().get("access")
    if not access:
        msg = "Label Studio token refresh returned no access token"
        raise RuntimeError(msg)
    return str(access)


def _fetch_ls_asset(ls_url: str, asset_path: str, access_token: str) -> bytes:
    import requests  # noqa: PLC0415

    split = urlsplit(asset_path)
    path = split.path
    if split.query:
        path += f"?{split.query}"
    if not path.startswith("/"):
        path = f"/{path}"

    response = requests.get(
        f"{ls_url}{path}",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=HTTP_ASSET_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.content


def _extract_pdf_path(task: dict[str, object]) -> str:
    data = task.get("data")
    if not isinstance(data, dict):
        msg = "Task data is missing"
        raise KeyError(msg)

    pdf = data.get("pdf")
    if isinstance(pdf, str) and pdf:
        return pdf

    for value in data.values():
        if isinstance(value, str) and (value.lower().endswith(".pdf") or "/data/upload/" in value):
            return value

    msg = f"No PDF path found in task data (keys: {list(data.keys())})"
    raise KeyError(msg)


def _filename_from_path(asset_path: str) -> str:
    return urlsplit(asset_path).path.rsplit("/", 1)[-1] or asset_path


def _set_task_text(
    ls_url: str,
    task_id: object,
    existing_data: dict[str, Any],
    pdf_path: str,
    text: str,
    access_token: str,
) -> None:
    import requests  # noqa: PLC0415

    response = requests.patch(
        f"{ls_url}/api/tasks/{task_id}",
        json={"data": {**existing_data, "pdf": pdf_path, "text": text}},
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=HTTP_PATCH_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
