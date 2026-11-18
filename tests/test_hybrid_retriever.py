from src.document_loader import file_loder
from src.ingestion import break_into_chunks
from src.vector_store import vector_store
from src.hybrid_retriever import hybrid_search

document = file_loder("data/raw/CNN.pdf")
chunks = break_into_chunks(document)

store = vector_store(chunks)

query = "What is a convolutional neural network?"

results = hybrid_search(chunks, store, query, k=3)

for i, result in enumerate(results, start=1):
    print(f"\n--- Hybrid Result {i} ---")
    print("Page:", result["Page_Number"])
    print("Hybrid Score:", round(result["Hybrid_Score"], 4))
    print(result["Text"])
