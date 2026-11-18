
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever:
    def __init__(self, chunks):
        if not chunks:
            raise ValueError("No chunks available for retrieval.")

        self.chunks = chunks

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english"
        )

        texts = [chunk["Text"] for chunk in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, k=3):
        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.tfidf_matrix
        ).flatten()

        top_indices = scores.argsort()[::-1][:k]

        results = []

        for index in top_indices:
            if scores[index] <= 0:
                continue

            results.append({
                "Page_Number": self.chunks[index]["Page_Number"],
                "Text": self.chunks[index]["Text"],
                "Score": float(scores[index])
            })

        return results
