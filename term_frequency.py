from collections import Counter

def compute_tf(tokens, vocab):
    """Computes the term frequency for each term in the vocabulary."""
    count = Counter(tokens)
    return { term: count[term] for term in vocab }  # Changed to raw counts