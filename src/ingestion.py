
import re
from collections import Counter


def tokenize_text(text):
    """
    NLP concept: Tokenization
    Splits text into words, numbers, and punctuation.
    """
    tokens = re.findall(r"\b\w+(?:['-]\w+)*\b|[^\w\s]", text.lower())
    return tokens


def generate_ngrams(tokens, n=2):
    """
    NLP concept: N-grams
    Generates sequences of n consecutive tokens.
    n=1: Unigrams
    n=2: Bigrams
    n=3: Trigrams
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    return [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]


def get_ngram_frequency(text, n=2):
    """
    Counts how frequently each N-gram occurs in the text.
    """
    tokens = tokenize_text(text)
    ngrams = generate_ngrams(tokens, n)
    return Counter(ngrams)


def break_into_chunks(document, size=200, overlap=100):
    """
    Splits extracted PDF text into overlapping chunks.
    Uses tokenization and N-gram processing for NLP analysis.
    """
    if size <= 0:
        raise ValueError("size must be greater than 0")

    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    chunks = []

    for doc in document:
        text = doc["Text"]
        page_no = doc["Page_Number"]

        # NLP concept: Tokenization
        tokens = tokenize_text(text)

        # NLP concept: N-gram frequency analysis
        bigram_frequency = get_ngram_frequency(text, n=2)

        # Preserve the original text-based chunking for RAG
        start = 0

        while start < len(text):
            end = start + size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_tokens = tokenize_text(chunk_text)

                chunks.append({
                    "Page_Number": page_no,
                    "Text": chunk_text,
                    "Tokens": chunk_tokens,
                    "Token_Count": len(chunk_tokens),
                    "Top_Bigrams": [
                        (" ".join(gram), count)
                        for gram, count in bigram_frequency.most_common(5)
                    ]
                })

            start += size - overlap

    return chunks
