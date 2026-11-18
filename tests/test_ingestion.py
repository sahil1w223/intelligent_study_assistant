
from src.document_loader import file_loder
from src.ingestion import (
    break_into_chunks,
    tokenize_text,
    generate_ngrams,
    get_ngram_frequency
)

document = file_loder("data/raw/CNN.pdf")

chunks = break_into_chunks(document)

print("Total chunks:", len(chunks))

print("\n--- Tokenization ---")
print(tokenize_text("CNN uses filters to extract features."))

print("\n--- Bigrams ---")
print(generate_ngrams(["cnn", "uses", "filters"], n=2))

print("\n--- Bigram Frequency ---")
print(get_ngram_frequency("CNN uses filters. CNN uses features.", n=2))

print("\n--- First Chunk ---")
if chunks:
    print("Page:", chunks[0]["Page_Number"])
    print("Text:", chunks[0]["Text"])
    print("Token count:", chunks[0]["Token_Count"])
    print("Top bigrams:", chunks[0]["Top_Bigrams"])
