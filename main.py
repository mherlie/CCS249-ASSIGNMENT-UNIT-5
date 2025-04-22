import wikipedia
from term_frequency import compute_tf
from tf_idf import compute_idf, compute_tfidf

# --- Wikipedia API Integration ---
topics = ["Artificial Intelligence", "Deep Learning", "Black Hole", "Newton's Law", "Orbital Resonance"]
documents = []

for topic in topics:
    try:
        documents.append(wikipedia.summary(topic, sentences=2))  # Get the summary (first 2 sentences)
    except wikipedia.exceptions.PageError:
        documents.append(f"Error: Could not retrieve Wikipedia page for '{topic}'")
    except wikipedia.exceptions.DisambiguationError as e:
        documents.append(f"Disambiguation Error for '{topic}': {e.options[0]}")

print("Documents from Wikipedia:")
for i, doc in enumerate(documents):
    print(f"Document {i + 1} ({topics[i]}): {doc}")

# Tokenize and lowercase the documents
tokenized_docs = [doc.lower().split() for doc in documents]

# Create a set of unique words (vocabulary)
vocabulary = sorted(set(word for doc in tokenized_docs for word in doc))

# Compute the term frequency for each document (RAW COUNTS)
tf_vectors = [compute_tf(doc, vocabulary) for doc in tokenized_docs]

print("\nTerm Frequency Vectors:")
for i, tf_vector in enumerate(tf_vectors):
    print(f"Document {i + 1}: {tf_vector}")

# Compute the IDF values
idf = compute_idf(tokenized_docs, vocabulary)

print("\nInverse Document Frequency:")
for term, idf_value in idf.items():
    print(f"{term}: {idf_value}")

# Compute TF-IDF vectors for each document
tfidf_vectors = [compute_tfidf(tf, idf, vocabulary) for tf in tf_vectors]

print("\nTF-IDF Vectors:")
for i, tfidf_vector in enumerate(tfidf_vectors):
    print(f"Document {i + 1}: {tfidf_vector}")
