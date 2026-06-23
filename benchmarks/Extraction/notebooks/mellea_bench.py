# %%
from pathlib import Path
from dotenv import load_dotenv
import sys


load_dotenv()
from mellea_lrc.extraction import extract_document_file

# %%
# Prints the sys.modules
import sys

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
    all_documet_path = Path.cwd().parent / "law-document-samples"
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

    sample_document = extract_document_file(file_path)
    return sample_document.text
