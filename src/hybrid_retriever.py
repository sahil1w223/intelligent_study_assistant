
from src.tfidf_retriever import TFIDFRetriever


def hybrid_search(chunks, store, query, k=3):
    
    tfidf_retriever = TFIDFRetriever(chunks)
    tfidf_results = tfidf_retriever.search(query, k=k)

   
    faiss_results = store.similarity_search_with_score(query, k=k)

  
    combined_results = {}

    for result in tfidf_results:
        key = (result["Page_Number"], result["Text"])
        combined_results[key] = {
            "Page_Number": result["Page_Number"],
            "Text": result["Text"],
            "TFIDF_Score": result["Score"],
            "FAISS_Score": None
        }

    for doc, score in faiss_results:
        page = doc.metadata["page"]
        text = doc.page_content
        key = (page, text)

        if key not in combined_results:
            combined_results[key] = {
                "Page_Number": page,
                "Text": text,
                "TFIDF_Score": 0.0,
                "FAISS_Score": float(score)
            }
        else:
            combined_results[key]["FAISS_Score"] = float(score)


    results = list(combined_results.values())

    tfidf_ranks = {
        (item["Page_Number"], item["Text"]): rank
        for rank, item in enumerate(tfidf_results, start=1)
    }

    faiss_ranks = {
        (doc.metadata["page"], doc.page_content): rank
        for rank, (doc, score) in enumerate(faiss_results, start=1)
    }

    for item in results:
        key = (item["Page_Number"], item["Text"])

        item["Hybrid_Score"] = (
            1 / (60 + tfidf_ranks[key])
            if key in tfidf_ranks else 0
        ) + (
            1 / (60 + faiss_ranks[key])
            if key in faiss_ranks else 0
        )

    results.sort(key=lambda item: item["Hybrid_Score"], reverse=True)

    return results[:k]
