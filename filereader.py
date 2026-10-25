from pathlib import Path
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader

# this read a single document
def _read_pdf_text(pdf_path: str | Path) -> str:
    """read and return the text content of a pdf file"""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    # Read the document from the path
    # Split the text in each pages as separate item in list
    # the join all together as string with "\n" as separator 
    reader = PdfReader(str(path))
    text_parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text_parts)

# this runs read_pdf_text() function for each document paths in the list
def read_multiple_pdfs(pdf_paths: List[str]) -> List[Document]:
    """read text from multiple pdfs and return a list of Documents object"""
    docs: List[Document] = []
    for path in pdf_paths:
        path_obj = Path(path)
        text = _read_pdf_text(path)
        # metadata={"source": str(path_obj)} stores the path as a string, since metadata should hold simple values.
        docs.append(Document(page_content = text, metadata = {"source": str(path_obj)}))

    return docs

# List all paths in the dataset
DOC_PATHS = [
    "./pdfs/climate_change.pdf",
    "./pdfs/global_warming.pdf"
]

# # ======================================================
# # Testing
# print(read_multiple_pdfs(DOC_PATHS))
# # ======================================================