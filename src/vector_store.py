from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


MODEL_NAME = "nomic-embed-text"


def vector_store(chunks, save_path="vector_store/faiss_index"):
    if not chunks:
        raise ValueError("No chunks found.")

    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    documents = [
        Document(
            page_content=chunk["Text"],
            metadata={"page": chunk["Page_Number"]}
        )
        for chunk in chunks
    ]

    store = FAISS.from_documents(
        documents,
        embeddings
    )

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    store.save_local(save_path)

    return store
