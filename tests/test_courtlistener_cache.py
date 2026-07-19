from __future__ import annotations

import io
import json
import os
import unittest
from unittest.mock import patch

from mellea_lrc.courtlistener.cache import CacheEntry, NullCache, R2Cache


class FakeR2Client:
    """In-memory stand-in for the S3-compatible calls used by Cloudflare R2."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def get_object(self, Bucket: str, Key: str) -> dict:
        payload = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(payload)}

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        self.objects[(Bucket, Key)] = Body
        self.content_type = ContentType


class R2CacheTests(unittest.TestCase):
    """Regression guards for env fallback and R2 object serialization."""

    def test_empty_bucket_uses_null_cache(self) -> None:
        """An unset bucket should disable caching without requiring R2 credentials."""
        with patch.dict(os.environ, {"R2_BUCKET": ""}, clear=False):
            self.assertIsInstance(R2Cache.from_env(), NullCache)

    def test_put_and_get_round_trip(self) -> None:
        """Cached envelopes should be written and read from one deterministic JSON object."""
        client = FakeR2Client()
        cache = R2Cache(bucket="cases", prefix="courtlistener/test", object_client=client)
        entry = CacheEntry(
            key="abc123",
            method="GET",
            endpoint="dockets/",
            params={},
            data={},
            status_code=200,
            url="https://example.test/dockets/",
            cached_at="2026-06-06T00:00:00Z",
            response={"id": 1},
        )

        cache.put(entry)
        result = cache.get("abc123")

        self.assertEqual(result, entry)
        self.assertIn(("cases", "courtlistener/test/abc123.json"), client.objects)
        self.assertEqual(client.content_type, "application/json")
        payload = json.loads(client.objects[("cases", "courtlistener/test/abc123.json")])
        self.assertEqual(payload["response"], {"id": 1})


if __name__ == "__main__":
    unittest.main()
