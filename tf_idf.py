from math import log

# Compute the Inverse Document Frequency
def compute_idf(tokenized_docs, vocab):
    N = len(tokenized_docs)
    idf_dict = {}
    for term in vocab:
        # Count the number of documents containing the term
        df = sum(term in doc for doc in tokenized_docs)
        # Compute IDF using the formula: idf(t) = log(N / df(t))
        idf_dict[term] = log(N / (df or 1))
    return idf_dict

# Compute TF-IDF
def compute_tfidf(tf_vector, idf, vocab):
    # Compute the TF-IDF score for each term in the vocabulary
    # using the formula: tf-idf(t, d) = tf(t, d) * idf(t)
    tfidf_vector = {}
    for term in vocab:
        tf = tf_vector.get(term, 0)  # Get TF, or 0 if term is not in document
        tfidf = tf * idf.get(term, 0)  # Multiply TF and IDF
        tfidf_vector[term] = tfidf
    return tfidf_vector