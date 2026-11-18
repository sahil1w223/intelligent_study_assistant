
from src.document_loader import file_loder
from src.ingestion import break_into_chunks
from src.vector_store import vector_store

documents = file_loder("data/raw/CNN_test_notes.pdf")
chunks = break_into_chunks(documents)

store = vector_store(chunks)

query = "What is the CNN?"
results = store.similarity_search(query, k=3)

for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Page:", result.metadata["page"])
    print(result.page_content)
