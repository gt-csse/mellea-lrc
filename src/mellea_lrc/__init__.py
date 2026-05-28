"""Hello package."""


def hello() -> str:
    """Hello placeholder."""
    return "Hello from mellea-lrc!"


from importlib.metadata import version  # noqa: E402


__version__ = version("mellea_lrc")
