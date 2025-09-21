# src/faiss_search.py

import os
import faiss
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
CLIP_EMBED_FILE = os.path.join(EMBEDDINGS_DIR, "clip_vectors.npy")
FAISS_INDEX_FILE = os.path.join(FAISS_DIR, "index.faiss")

if not os.path.exists(FAISS_DIR):
    os.makedirs(FAISS_DIR)

# -----------------------------
# Load CLIP model (for queries)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# -----------------------------
# Build FAISS Index
# -----------------------------
def build_faiss_index(embeddings):
    """
    embeddings: np.array of shape (num_images, embedding_dim)
    returns: faiss index
    """
    embedding_dim = embeddings.shape[1]
    # FAISS IndexFlatIP for cosine similarity (use normalized embeddings)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index


def save_faiss_index(index, file_path=FAISS_INDEX_FILE):
    faiss.write_index(index, file_path)
    print(f"FAISS index saved to {file_path}")


def load_faiss_index(file_path=FAISS_INDEX_FILE):
    index = faiss.read_index(file_path)
    print(f"FAISS index loaded. Total vectors: {index.ntotal}")
    return index


# -----------------------------
# Search Function
# -----------------------------
def search_images(query_text, index, filenames, top_k=5):
    """
    query_text: string
    index: FAISS index
    filenames: list of image filenames
    top_k: number of results to return
    """
    # Encode query
    inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        # Normalize
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()

    # Search
    distances, indices = index.search(text_features, top_k)
    results = [filenames[i] for i in indices[0]]
    return results, distances[0]


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    from ingestion import load_and_preprocess_images
    from embeddings import generate_clip_embeddings

    # Load images and embeddings
    images, filenames = load_and_preprocess_images()
    embeddings = generate_clip_embeddings(images)
    # Build FAISS index
    index = build_faiss_index(embeddings)
    save_faiss_index(index)

    # Load index and run a sample query
    index = load_faiss_index()
    query = "beach"
    results, scores = search_images(query, index, filenames, top_k=5)
    print("Top results:", results)
    print("Similarity scores:", scores)
