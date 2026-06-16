"""Practice Mella."""

# %%
from dotenv import load_dotenv
from pathlib import Path

import mellea
from mellea.stdlib.components.docs.richdocument import RichDocument

# %%
load_dotenv()
m = mellea.start_session()

# %%
email = m.instruct("What are case citations in legal documents? How are they formatted?")

# %%
print(email)


# %%
# Get the path to a PDF file
samples_path = Path.cwd().parent / "law-document-samples" / "federal"

if samples_path.exists() and samples_path.is_dir():
    print(samples_path, "exists and is a dir")
else:
    print(samples_path, "doesn't exists or is not a dir")

file_path = next(samples_path.iterdir())

if file_path.exists() and file_path.is_file() and file_path.suffix == ".pdf":
    print(file_path, "exists, is a file, and is a pdf")
else:
    print(file_path, "doesn't exists, is not a file, or is not a pdf")
# %%
# Extract case citations
## Convert to plain-text
document = RichDocument.from_document_file(file_path)


# %%
citations = m.instruct(
    "return a list of all **case** citations in the document. keep order as they appear in the text. include short, supra, and id citations. text: {{text}}",
    user_variables={"text": document.to_markdown()},
)

# %%
print(citations)

# %%
