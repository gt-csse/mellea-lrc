"""Extract case citations and label them using different tools."""

from mellea_lrc.extraction.base import BaseExtractor


class Extractor:
    """Extract case citations and label them."""

    def __init__(self, tool: BaseExtractor):  # noqa: ANN204
        """Initialize extractor with tool, e.g., Mellea, Eyecite, or Hybrid."""
        self.tool = tool
