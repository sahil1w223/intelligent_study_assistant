import pymupdf
from pathlib import Path

def file_loder(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{file_path} is Not Exist")

    document = []

    with pymupdf.open(path) as w:
        for page_number, content in enumerate(w, start=1):
            text = content.get_text("text").strip()

            if text:
                document.append({
                    "Page_Number": page_number,
                    "Text": text
                })

    return document
