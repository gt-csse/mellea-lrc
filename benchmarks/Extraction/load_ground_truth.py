"""Load the ground truth files."""

# %%
import json
from pathlib import Path
from collections import defaultdict
import hashlib

from mellea_lrc.extraction import ExtractedDocument, ExtractionBackend, ExtractionMetadata, ExtractedCitation
from mellea_lrc.preprocessing import PreprocessingMetadata, PreprocessingBackend
from mellea_lrc.core import (
    SourceMetadata,
    SourceFormat,
    FullCaseCitation,
    ShortCaseCitation,
    FullJournalCitation,
    FullLawCitation,
    SupraCitation,
    IdCitation,
    ReferenceCitation,
    UnknownCitation,
    CitationKind,
    Span,
)

from generate_test_cases import save_citations

# %%
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


def load_lb(citation_data: list) -> ExtractedDocument:
    """Deserialize the Label Studio exported data."""
    # Extract the `citations`
    citation_result: list = citation_data[0].get("annotations", [])[0].get("prediction").get("result", [])
    citation_groupings = defaultdict(list)
    citations_with_ids = filter(lambda item: "id" in item, citation_result)
    # citation_relations = filter(lambda item: "id" not in item, citation_result) # noqa: ERA001 # TODO: need to create reference map
    citations_by_type = filter(lambda item: item["type"] == "labels", citations_with_ids)
    citations_with_ids = filter(lambda item: "id" in item, citation_result)
    for item in citations_with_ids:
        citation_groupings[item["id"]].append(item)

    # Build the CanonicalCitions
    citations = []
    for item in citations_by_type:
        citation_id = item["id"]
        # Get all items with the same id:
        grouping = citation_groupings[citation_id]
        # Get the universally shared information (all the info shared by an ID)
        basic_info = item["value"]
        start, end = basic_info["start"], basic_info["end"]
        matched_text = basic_info["text"]
        kind = CitationKind(basic_info["labels"][0])
        span = Span(start, end)
        # Collect all Citation information
        citation_parameters = {}
        citation_fields = filter(lambda val: val["type"] == "textarea", grouping)
        for element in citation_fields:
            if element["from_name"] not in citation_parameters:
                citation_parameters[element["from_name"]] = None
            citation_parameters[element["from_name"]] = element["value"]["text"][0] or None
        # Build the CanonicalCitation
        citation = mapping[kind](**citation_parameters)
        # Build `ExtractedCitation`
        extracted_citation = ExtractedCitation(
            citation_id=citation_id, span=span, matched_text=matched_text, citation=citation
        )
        citations.append(extracted_citation)
    # Convert to a tuple
    citations = tuple(citations)
    # Extract `SourceMetadata`
    source_dc = SourceMetadata(SourceFormat.PDF)
    # Extract `ExtractedMetadata`
    extracted_dc = ExtractionMetadata(backend=ExtractionBackend.MANUAL)
    # Extract `PreprocessingMetadata`
    metadata_dc = PreprocessingMetadata(backend=PreprocessingBackend.DOCLING)
    # Extract the `text` section
    text: str = citation_data[0].get("data", {}).get("text", "")
    return ExtractedDocument(
        citations=citations,
        text=text,
        preprocessing_metadata=metadata_dc,
        extraction_metadata=extracted_dc,
        source_metadata=source_dc,
    )


# %%
def main() -> int:
    """Load and deserialize Label Studio's output."""

    # Ground-truth file path
    dir_path = Path(__file__).parent / "data" / "ground_truth"
    if not (dir_path.exists() and dir_path.is_dir()):
        msg = f"The directory doesn't exists: {dir_path}"
        raise Exception(msg)

    # Load the file
    file_path = next(dir_path.iterdir())
    file_content: list = []
    with file_path.open("r") as file:
        file_content = json.load(file)
    raw_file = file_path.read_bytes()
    file_hash = hashlib.sha256(raw_file).hexdigest()
    cached_files_path = dir_path / ".cache"
    cached_files_path.mkdir(exist_ok=True)
    hashed_file_name = cached_files_path / f"{file_hash}.json"

    citation = load_lb(file_content)
    save_citations(hashed_file_name, citation)
    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())

# %%
