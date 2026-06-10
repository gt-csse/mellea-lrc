"""Extract case citations and label them."""

import abc


class BaseExtractor(abc.ABC):
    """Abstract base class for extractors."""

    @abc.abstractmethod
    def get_citations() -> None:
        """Identify, retrieve, and classify case law citations."""
