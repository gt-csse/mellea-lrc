"""Run benchmarks on Mellea models."""

# %%
import copy
import hashlib
from pathlib import Path
from dataclasses import asdict
import json

from mellea_lrc.extraction import (
    extract_document_file,
    extract_citations,
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


# %%
# Get a PDF document
def get_document() -> Path:
    """Get a PDF file from the samples folder."""
    all_documet_path = Path(__file__).parent / "data" / "federal"
    if all_documet_path.exists() and all_documet_path.is_dir():
        document_path = next(all_documet_path.iterdir())
        if not (document_path.exists() or document_path.is_file()):
            msg = f"{document_path} doesn't exist or it is not a file"
            raise Exception(msg)
    else:
        msg = f"{all_documet_path} doesn't exists or it isn't a directory"
        raise Exception(msg)
    return document_path


# %%
# Convert PDFs into plain text
def convert_to_text(file_path: Path) -> str:
    """Convert PDF to text and return it as a string."""
    sample_document = extract_document_file(file_path)
    return sample_document.text


# %%
# Save the citations
def save_citations(path: Path, document: ExtractedDocument) -> None:
    """Save the citions."""
    document_dict = asdict(document)
    # Add `kind` key to each Citation
    kind_mappings = {item.citation_id: item.citation.kind for item in document.citations}
    for citation in document_dict.get("citations", []):
        cite_id = citation["citation_id"]
        citation["citation"]["kind"] = kind_mappings[cite_id].value
    with path.open("w", encoding="utf-8") as file:
        json.dump(document_dict, file, indent=4)


# %%
# Load the saved citations
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


def load_citations(path: Path) -> ExtractedDocument:
    """Load the saved citations."""
    data = None
    # Load file into data
    if path.exists() and path.is_file() and path.suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        msg = f"{path} doesn't exists, is not a file, or it isn't json"
        raise Exception(msg)

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

            temp = ExtractedCitation(citation=citation_type, span=span_dc, **citation_copy)
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


# %%
def main() -> int:
    """Run the mellea benchmark."""
    # Get a PDF to analyze
    file_path = get_document()
    # Convert the file to text and save a copy
    dir_name = Path(__file__).parent / ".cache"
    dir_name.mkdir(exist_ok=True)
    raw = file_path.read_bytes()
    hash_string = hashlib.sha256(raw).hexdigest()
    text_path = dir_name / str(hash_string + ".txt")
    text = ""
    if hash_string in [path.stem for path in dir_name.iterdir()]:
        text = text_path.read_text(encoding="utf-8")
    else:
        text = convert_to_text(file_path)
        text_path.write_text(text)

    citations: ExtractedDocument = extract_citations(text)
    saved_document_path = dir_name / str(hash_string + ".json")
    save_citations(saved_document_path, citations)
    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())
# %%
