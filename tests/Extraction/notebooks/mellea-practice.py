"""Practice Mella."""

# %%
from dotenv import load_dotenv
from pathlib import Path

# Ensure that Ollama's host name is updated
load_dotenv()
from mellea_lrc.extraction import MelleaExtractor  # noqa: E402


# %%
# Get the path to a PDF file
samples_path = Path.cwd().parent / "law-document-samples" / "federal"

if samples_path.exists() and samples_path.is_dir():
    print(samples_path, "exists and is a dir")  # noqa: T201
else:
    print(samples_path, "doesn't exists or is not a dir")  # noqa: T201

file_path = next(samples_path.iterdir())

if file_path.exists() and file_path.is_file() and file_path.suffix == ".pdf":
    print(file_path, "exists, is a file, and is a pdf")  # noqa: T201
else:
    print(file_path, "doesn't exists, is not a file, or is not a pdf")  # noqa: T201

# %%
text = MelleaExtractor.extract_structured_text(file_path)

# %%
citations = MelleaExtractor.extract_citations(text)
# %%
print(file_path.name)  # noqa: T201
for index, citation in enumerate(start=1, iterable=citations):
    print(f"{index}: {citation}")  # noqa: T201

# %%
