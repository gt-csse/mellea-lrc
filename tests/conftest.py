"""Shared pytest configuration."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register opt-in remote smoke test arguments."""
    parser.addoption(
        "--run-remote-smoke",
        action="store_true",
        default=False,
        help="Run smoke tests that call external services.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip remote smoke tests unless explicitly requested."""
    if config.getoption("--run-remote-smoke"):
        return

    skip_remote_smoke = pytest.mark.skip(reason="Pass --run-remote-smoke to run remote smoke tests.")
    for item in items:
        if "remote_smoke" in item.keywords:
            item.add_marker(skip_remote_smoke)
