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
    parser.addoption(
        "--run-llm-evaluations",
        action="store_true",
        default=False,
        help="Run live evaluations against the configured LLM.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip externally dependent tests unless explicitly requested."""
    skip_remote_smoke = pytest.mark.skip(reason="Pass --run-remote-smoke to run remote smoke tests.")
    skip_llm_evaluation = pytest.mark.skip(
        reason="Pass --run-llm-evaluations to run live LLM evaluations."
    )
    for item in items:
        if "remote_smoke" in item.keywords and not config.getoption("--run-remote-smoke"):
            item.add_marker(skip_remote_smoke)
        if "llm_evaluation" in item.keywords and not config.getoption("--run-llm-evaluations"):
            item.add_marker(skip_llm_evaluation)
