"""Run benchmarks on Mellea models."""

# %%
import hashlib
from pathlib import Path
import sys
from dataclasses import asdict
import json
from typing import cast

from mellea_lrc.extraction import extract_document_file, extract_citations, DocumentExtraction, ExtractedCitation
from mellea_lrc.core import CanonicalCitation, FullCaseCitation, FullLawCitation, FullJournalCitation, ShortCaseCitation, SupraCitation, IdCitation
from mellea_lrc.preprocessing import PreprocessedDocument, PreprocessedDocumentMetadata

# %%
# Prints the sys.modules
print(type(sys.modules))
for value in sys.modules:
    if "mellea" in value:
        print(value)

# %%
for index, value in enumerate(sys.path, start=1):
    print(f"{index}: {value!r}")


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
def save_citations(path: Path, document: DocumentExtraction) -> None:
    """Save the citions."""
    document_dict = asdict(document)
    # Add `kind` key to each Citation
    for idx, citation in enumerate(document_dict.get("citations", [])):
        citation["citation"]["kind"] = document.citations[idx].citation.kind
    with path.open("w", encoding="utf-8") as file:
        json.dump(document_dict, file, indent=4)

# %%
# Load the saved citations
mapping = { citation_type.kind : citation_type for citation_type in (FullCaseCitation, FullJournalCitation, FullLawCitation, ShortCaseCitation, SupraCitation, IdCitation)}
def load_citations(path: Path) -> DocumentExtraction:
    """Load the saved citations."""
    data = None
    if path.exists() and path.is_file() and path.suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        msg = f"{path} doesn't exists, is not a file, or it isn't json"
        raise Exception(msg)
    citations = []
    if data:
        for citation in data.get("citations", []):
            sub_citation = citation.pop("citation")
            kind =  sub_citation.pop("kind") # raise error if key is missing (should be there)
            citation_type = mapping[kind](**sub_citation) # Create a citation of 'kind' type
            temp = ExtractedCitation(citation=citation_type, **citation)
            citations.append(temp)
        preprocessed = data.pop("preprocessed")
        pre_pros_meta = PreprocessedDocumentMetadata(**preprocessed["metadata"])
        pre_pros_doc = PreprocessedDocument(text=preprocessed["text"], metadata=pre_pros_meta)

        citations = cast(tuple[ExtractedCitation, ...], citations)
        document = DocumentExtraction(preprocessed=pre_pros_doc, citations=citations)
        return document
    else:
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
    if hash_string in list(path.stem for path in dir_name.iterdir()):
        text = text_path.read_text(encoding="utf-8")
    else:
        text = convert_to_text(file_path)
        text_path.write_text(text)

    header = "\n".join(text.split("\n")[:9])
    print(header)

    citations: DocumentExtraction = extract_citations(text)
    saved_document_path = dir_name / str(hash_string + ".json")
    save_citations(saved_document_path, citations)
    citation = load_citations(saved_document_path)
    #assert(citation == citations)
    separator = "-" * 80
    if citations.text != citation.text:
        print("The text are not equal")
    if citations.preprocessed != citation.preprocessed:
        print("the preprocessed are not equal!")
        if citations.preprocessed.metadata != citation.preprocessed.metadata:
            print(" - The metadata are not equal")
            print()
            x = json.dumps(asdict(citations.preprocessed.metadata), indent=4)
            print(x)
            print(separator)
            print()
            x = json.dumps(asdict(citation.preprocessed.metadata), indent=4)
            print(x)
            print(separator)
            



    if citations.citations != citation.citations:
        print("the citations are not equal")
    return 0

# %%

if __name__ == "__main__":
    raise SystemExit(main())

# %%
