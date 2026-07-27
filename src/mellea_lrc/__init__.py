"""Hello package."""


def hello() -> str:
    """Hello placeholder."""
    return "Hello from mellea-lrc!"


from importlib.metadata import version


__version__ = version("mellea_lrc")
