"""Convert PDF files into plain-text with Docling."""

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.document import ConversionResult


def _get_files() -> list[Path]:
    """Return a list of PDF from the data folder.

    Return:
        A list of file paths to PDF

    Raises:
        An exception of path doesn't exists

    """
    data_path = Path(__file__).parent.parent.parent / "tests/data/federal"  # Hardcoding for now
    if not data_path.exists():
        message = f"{data_path} doesn't exists"
        raise Exception(message)
    return [
        file_path.resolve()
        for file_path in data_path.iterdir()
        if file_path.is_file() and file_path.suffix == ".pdf"
    ]


def main() -> None:
    """Convert PDFs into plaint-text."""
    pdf_paths = _get_files()

    pass


if __name__ == "__main__":
    main()
