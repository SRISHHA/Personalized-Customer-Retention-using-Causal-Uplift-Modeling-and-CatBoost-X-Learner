from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_documents(doc_folder="docs"):
    text = ""

    folder = Path(doc_folder)

    for txt_file in folder.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            text += f.read() + "\n"

    return text
def chunk_text(text, chunk_size=300, overlap=50):
    """
    Split text into overlapping chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks
def build_vector_store(chunks):
    """
    Convert chunks into embeddings and build a FAISS index.
    """

    embeddings = embedding_model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, chunks

def retrieve_context(question, index, chunks, top_k=2):
    """
    Retrieve the most relevant chunks for a given question.
    """

    question_embedding = embedding_model.encode([question])

    question_embedding = np.array(question_embedding).astype("float32")

    distances, indices = index.search(question_embedding, top_k)

    retrieved = []

    for idx in indices[0]:
        retrieved.append(chunks[idx])

    return "\n\n".join(retrieved)
