
from src.document_loader import file_loder
from src.ingestion import break_into_chunks

pdf_path = "data/raw/CNN_test_notes.pdf"

documents = file_loder(pdf_path)
chunks = break_into_chunks(documents)

print("Pages extracted:", len(documents))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {i} | Page_Number {chunk['Page_Number']} ---")
    print(chunk["Text"])