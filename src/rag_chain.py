
from langchain_ollama import ChatOllama
from src.hybrid_retriever import hybrid_search


def get_answer(store, chunks, question):
    # Retrieve relevant passages using TF-IDF + FAISS
    relevant_docs = hybrid_search(
        chunks=chunks,
        store=store,
        query=question,
        k=3
    )

    if not relevant_docs:
        return {
            "answer": "I could not find the answer in your notes.",
            "sources": []
        }

    # Build context for the LLM
    context = "\n\n".join(
        f"Page {doc['Page_Number']}: {doc['Text']}"
        for doc in relevant_docs
    )

    llm = ChatOllama(
        model="qwen2.5:0.5b",
        temperature=0
    )

    prompt = f"""
Answer the question using only the provided context.

If the answer is not present in the context, say:
"I could not find the answer in your notes."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": relevant_docs
    }
