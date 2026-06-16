"""Cache backends for CourtListener access service responses."""

# ruff: noqa: ANN401, ARG002, D101, D102, EM101, PLC0415, TRY003

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class CacheEntry:
    key: str
    method: str
    endpoint: str
    params: dict[str, Any]
    data: dict[str, Any]
    status_code: int
    url: str
    cached_at: str
    response: Any

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CacheEntry:
        return cls(
            key=payload["key"],
            method=payload["method"],
            endpoint=payload["endpoint"],
            params=payload.get("params") or {},
            data=payload.get("data") or {},
            status_code=int(payload["status_code"]),
            url=payload["url"],
            cached_at=payload["cached_at"],
            response=payload["response"],
        )


class CacheStore(Protocol):
    def get(self, key: str) -> CacheEntry | None: ...

    def put(self, entry: CacheEntry) -> None: ...


class NullCache:
    def get(self, key: str) -> None:
        return None

    def put(self, entry: CacheEntry) -> None:
        return None


class R2Cache:
    def __init__(
        self,
        bucket: str,
        prefix: str = "courtlistener/v4",
        object_client: Any = None,
    ) -> None:
        if object_client is None:
            import boto3

            object_client = boto3.client(
                "s3",
                endpoint_url=_r2_endpoint_url(),
                aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID") or None,
                aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY") or None,
                region_name=os.getenv("R2_REGION", "auto"),
            )
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = object_client

    @classmethod
    def from_env(cls) -> R2Cache | NullCache:
        bucket = os.getenv("R2_BUCKET", "").strip()
        if not bucket:
            return NullCache()
        return cls(bucket=bucket, prefix=os.getenv("R2_PREFIX", "courtlistener/v4"))

    def get(self, key: str) -> CacheEntry | None:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self._object_key(key))
        except Exception as exc:
            if _is_missing_object_key(exc):
                return None
            raise
        return CacheEntry.from_dict(json.loads(obj["Body"].read().decode("utf-8")))

    def put(self, entry: CacheEntry) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._object_key(entry.key),
            Body=json.dumps(entry.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    def _object_key(self, key: str) -> str:
        return f"{self.prefix}/{key}.json"


def _is_missing_object_key(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    code = response.get("Error", {}).get("Code")
    return code in {"NoSuchKey", "404", "NotFound"}


def _r2_endpoint_url() -> str:
    endpoint_url = os.getenv("R2_ENDPOINT_URL", "").strip()
    if endpoint_url:
        return endpoint_url

    account_id = os.getenv("R2_ACCOUNT_ID", "").strip()
    if account_id:
        return f"https://{account_id}.r2.cloudflarestorage.com"

    raise RuntimeError("Set R2_ENDPOINT_URL or R2_ACCOUNT_ID when R2_BUCKET is configured.")
