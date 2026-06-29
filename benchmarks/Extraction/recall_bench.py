"""Run benchmarks on Mellea models."""

# %%
import hashlib
from pathlib import Path
import sys

from mellea_lrc.extraction import extract_document_file, extract_citations, DocumentExtraction

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

    # upload_schema.main()
    # upload_tasks.main(paths=[str(text_path)]) Don't upload them to Label Studio
    citations: DocumentExtraction = extract_citations(text)

    

    return 0


# %%

if __name__ == "__main__":
    raise SystemExit(main())

# %%
