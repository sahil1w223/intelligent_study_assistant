from src.document_loader import file_loder
from src.ingestion import break_into_chunks
from src.tfidf_retriever import TFIDFRetriever

document = file_loder("data/raw/CNN.pdf")
chunks = break_into_chunks(document)

retriever = TFIDFRetriever(chunks)

results = retriever.search("What is a convolutional neural network?", k=3)

for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Page:", result["Page_Number"])
    print("Score:", round(result["Score"], 3))
    print(result["Text"])
