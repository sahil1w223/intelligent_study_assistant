
from src.document_loader import file_loder
from src.ingestion import break_into_chunks
from src.rag_chain import get_answer

documents = file_loder("data/raw/CNN_test_notes.pdf")
chunks = break_into_chunks(documents)

question = "What does ReLU do?"

result = get_answer(chunks, question)

print("\nAnswer:")
print(result["answer"])

print("\nSources:")
for doc in result["sources"]:
    print("Page:", doc.metadata["page"])
    print(doc.page_content[:300])
