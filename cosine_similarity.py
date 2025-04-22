import wikipedia
import math
from collections import Counter

def get_documents_from_wikipedia(topics, sentences=2):
    """Fetches summaries from Wikipedia for a list of topics."""
    documents = []
    for topic in topics:
        try:
            documents.append(wikipedia.summary(topic, sentences=sentences))
        except wikipedia.exceptions.PageError:
            documents.append(f"Error: Could not retrieve Wikipedia page for '{topic}'")
        except wikipedia.exceptions.DisambiguationError as e:
            documents.append(f"Disambiguation Error for '{topic}': {e.options[0]}")
    return documents

def compute_tf(tokens, vocab):
    """Computes the raw term frequency for each term in the vocabulary."""
    count = Counter(tokens)
    return {term: count[term] for term in vocab}

def compute_idf(tokenized_docs, vocab):
    """Computes the Inverse Document Frequency (IDF) for each term."""
    N = len(tokenized_docs)
    idf_dict = {}
    for term in vocab:
        df = sum(term in doc for doc in tokenized_docs)
        idf_dict[term] = math.log(N / (df or 1))
    return idf_dict

def compute_tfidf(tf_vector, idf, vocab):
    """Computes the TF-IDF score for each term in a document."""
    tfidf_vector = {}
    for term in vocab:
        tf = tf_vector.get(term, 0)
        tfidf = tf * idf.get(term, 0)
        tfidf_vector[term] = tfidf
    return tfidf_vector

def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    common_terms = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in common_terms)
    magnitude_vec1 = math.sqrt(sum(vec1.get(term, 0) ** 2 for term in vec1))
    magnitude_vec2 = math.sqrt(sum(vec2.get(term, 0) ** 2 for term in vec2))
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    return dot_product / (magnitude_vec1 * magnitude_vec2)

if __name__ == "__main__":
    topics = ["Artificial Intelligence", "Deep Learning", "Black Hole", "Newton's Law", "Orbital Resonance"]  
    documents = get_documents_from_wikipedia(topics, sentences=2)

    print("Documents:")
    for i, doc in enumerate(documents):
        print(f"Document {i + 1} ({topics[i]}): {doc}")

    tokenized_docs = [doc.lower().split() for doc in documents]
    vocabulary = sorted(set(word for doc in tokenized_docs for word in doc))

    tf_vectors = [compute_tf(doc, vocabulary) for doc in tokenized_docs]
    idf = compute_idf(tokenized_docs, vocabulary)
    tfidf_vectors = [compute_tfidf(tf, idf, vocabulary) for tf in tf_vectors] # Corrected line

    print("\nCosine Similarity Matrix:")
    num_docs = len(documents)
    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            similarity = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j])
            print(f"Doc {i + 1} vs. Doc {j + 1}: {similarity:.4f}")