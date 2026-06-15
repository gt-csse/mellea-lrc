"""LLM-based extraction augmentation (placeholder)."""

from mellea_lrc.extraction.result import DocumentExtraction
from mellea_lrc.preprocessing.document import PreprocessedDocument


class LLMExtractionAugmenter:
    """Use an LLM to recover missed spans or repair parsed citation fields."""

    def augment(
        self,
        preprocessed: PreprocessedDocument,
        baseline: DocumentExtraction,
    ) -> DocumentExtraction:
        """Augment baseline extraction with LLM-proposed citations or field fixes."""
        _ = preprocessed
        _ = baseline
        msg = "LLMExtractionAugmenter is not implemented yet."
        raise NotImplementedError(msg)
