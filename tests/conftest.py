"""Shared pytest configuration."""

from __future__ import annotations

from collections.abc import Callable

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register opt-in remote smoke test arguments."""
    parser.addoption(
        "--run-heavy",
        action="store_true",
        default=False,
        help="Run heavy local integration tests, such as real Docling conversion.",
    )
    parser.addoption(
        "--run-remote-smoke",
        action="store_true",
        default=False,
        help="Run smoke tests that call external services.",
    )
    parser.addoption(
        "--courtlistener-url",
        action="store",
        default=None,
        help="Base URL for a deployed CourtListener access service.",
    )
    parser.addoption(
        "--remote-timeout",
        action="store",
        default=30.0,
        type=float,
        help="Timeout in seconds for remote smoke test HTTP calls.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip opt-in tests unless explicitly requested."""
    skip_heavy = pytest.mark.skip(reason="Pass --run-heavy to run heavy tests.")
    skip_remote_smoke = pytest.mark.skip(reason="Pass --run-remote-smoke to run remote smoke tests.")
    for item in items:
        if "heavy" in item.keywords and not config.getoption("--run-heavy"):
            item.add_marker(skip_heavy)
        if "remote_smoke" in item.keywords and not config.getoption("--run-remote-smoke"):
            item.add_marker(skip_remote_smoke)


@pytest.fixture
def remote_timeout(request: pytest.FixtureRequest) -> float:
    """Return the configured remote smoke test timeout."""
    return float(request.config.getoption("--remote-timeout"))


def _remote_url_fixture(option_name: str, service_name: str) -> Callable[[pytest.FixtureRequest], str]:
    def fixture(request: pytest.FixtureRequest) -> str:
        value = request.config.getoption(option_name)
        if not value:
            pytest.skip(f"Pass {option_name} to run {service_name} remote smoke tests.")
        return str(value).rstrip("/")

    return fixture


courtlistener_url = pytest.fixture(
    _remote_url_fixture("--courtlistener-url", "CourtListener"),
)
