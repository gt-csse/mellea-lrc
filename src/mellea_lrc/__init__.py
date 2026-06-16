"""Mellea legal retrieval and citation helpers."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("mellea-lrc")
except PackageNotFoundError:
    __version__ = "0+unknown"


def hello() -> str:
    """Hello placeholder."""
    return "Hello from mellea-lrc!"
