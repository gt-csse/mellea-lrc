"""Functions to serialize and deserialize the extracted citations."""

import json
from pathlib import Path
import copy
from dataclasses import asdict

from mellea_lrc.extraction import (
    ExtractedCitation,
    ExtractedDocument,
    ExtractionMetadata,
    ExtractionBackend,
)
from mellea_lrc.core import (
    FullCaseCitation,
    FullLawCitation,
    FullJournalCitation,
    ShortCaseCitation,
    SupraCitation,
    IdCitation,
    Span,
    SourceMetadata,
    SourceFormat,
    UnknownCitation,
    ReferenceCitation,
    CitationKind,
)
from mellea_lrc.preprocessing import PreprocessingMetadata, PreprocessingBackend

mapping = {
    CitationKind(citation_type.kind): citation_type
    for citation_type in (
        FullCaseCitation,
        FullJournalCitation,
        FullLawCitation,
        ShortCaseCitation,
        SupraCitation,
        IdCitation,
        UnknownCitation,
        ReferenceCitation,
    )
}


# Save the citations
def save_citations(path: Path, document: ExtractedDocument) -> None:
    """Save the citions.

    Args:
    ----
        path (Path): The path to save the citations.
        document (ExtractedDocument): The extracted document to save.

    Returns:
    -------
        None

    Raises:
    ------
        Exception: If the document cannot be saved.

    """

    document_dict = asdict(document)
    # Add `kind` key to each Citation
    kind_mappings = {item.citation_id: item.citation.kind for item in document.citations}
    for citation in document_dict.get("citations", []):
        cite_id = citation["citation_id"]
        citation["citation"]["kind"] = kind_mappings[cite_id].value
    with path.open("w", encoding="utf-8") as file:
        json.dump(document_dict, file, indent=4)


# Load the citations
def load_citations(path: Path) -> ExtractedDocument:
    """Load the saved citations.

    Args:
    ----
        path (Path): The path to load the citations from.

    Returns:
    -------
        ExtractedDocument: The extracted document with citations.

    Raises:
    ------
        Exception: If the document cannot be loaded.

    """

    data = None
    # Load file into data
    if path.exists() and path.is_file() and path.suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        msg = f"{path} doesn't exists, is not a file, or it isn't json"
        raise FileNotFoundError(msg)

    # Reconstruct dataclasses
    if data:
        citations = []
        # Reconstruct all of the `ExtractedCitation` dataclasses
        for citation in data.pop("citations", []):
            citation_copy = copy.deepcopy(citation)
            # Reconstruct `CanonicalCitation`
            canonical_citation = citation_copy.pop("citation")
            kind = canonical_citation.pop("kind")  # raise error if key is missing (should be there)
            citation_type = mapping[CitationKind(kind)](
                **canonical_citation
            )  # Create a citation of 'kind' type
            # Reconstruct `Span`
            span = citation_copy.pop("span")
            span_dc = Span(**span)
            locator_span = citation_copy.pop("locator_span")
            locator_span_dc = Span(**locator_span)

            temp = ExtractedCitation(
                citation=citation_type, span=span_dc, locator_span=locator_span_dc, **citation_copy
            )
            citations.append(temp)
        citations = tuple(citations)

        # Reconstruct `PreprocessingMetadata`
        preprocessed = data.pop("preprocessing_metadata")
        backend = preprocessed.pop("backend")
        backend_dc = PreprocessingBackend(backend)
        preprocessing_metadata = PreprocessingMetadata(backend=backend_dc, **preprocessed)

        # Reconstruct `ExtractedBackend`
        extraction_meta = data.pop("extraction_metadata")
        extraction_backend = extraction_meta.pop("backend")
        extraction_backend_dc = ExtractionBackend(extraction_backend)
        extraction_meta_dc = ExtractionMetadata(backend=extraction_backend_dc, **extraction_meta)

        # Reconstruct `SourceMetadata`
        source_meta = data.pop("source_metadata")
        format_data = source_meta.pop("format")
        format_data_dc = SourceFormat(format_data)
        source_meta_dc = SourceMetadata(format=format_data_dc, **source_meta)

        return ExtractedDocument(
            source_metadata=source_meta_dc,
            citations=citations,
            preprocessing_metadata=preprocessing_metadata,
            extraction_metadata=extraction_meta_dc,
            **data,
        )
    msg = f"Couldn't create DocumentExtraction for {path}"
    raise Exception(msg)
