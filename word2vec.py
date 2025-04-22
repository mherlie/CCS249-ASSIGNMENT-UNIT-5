import wikipedia
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

# --- Step 1: Get Wikipedia summaries ---
topics = ["Artificial Intelligence", "Deep Learning", "Black Hole", "Newton's Law", "Orbital Resonance"]
documents = []

for topic in topics:
    try:
        documents.append(wikipedia.summary(topic, sentences=2))
    except wikipedia.exceptions.PageError:
        documents.append(f"Error: Could not retrieve Wikipedia page for '{topic}'")
    except wikipedia.exceptions.DisambiguationError as e:
        documents.append(f"Disambiguation Error for '{topic}': {e.options[0]}")

# --- Step 2: Tokenize and preprocess ---
tokenized_docs = [doc.lower().split() for doc in documents]

# --- Step 3: Train Word2Vec model ---
model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

# --- Step 4: Convert documents to average word vectors ---
def get_doc_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

doc_vectors = np.array([get_doc_vector(doc, model) for doc in tokenized_docs])

# --- Step 5: Prepare labels and train classifier ---
labels = list(range(len(documents)))  # Label: 0 to 4 for each topic

classifier = LogisticRegression(max_iter=1000)
classifier.fit(doc_vectors, labels)

# --- Step 6: Predict and evaluate ---
predictions = classifier.predict(doc_vectors)

print("Logistic Regression Classification Report:\n")
print(classification_report(labels, predictions, zero_division=1))
