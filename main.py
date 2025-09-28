# main.py

import os
import numpy as np
from PIL import Image
import faiss
from src.ingestion import load_and_preprocess_images
from src.faiss_search import load_faiss_index, search_images

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings/clip_vectors.npy")
FAISS_INDEX_FILE = os.path.join(BASE_DIR, "faiss_index/index.faiss")

# -----------------------------
# Load images
# -----------------------------
try:
    images, filenames = load_and_preprocess_images(save_processed=False)  # already processed
    if len(images) == 0:
        raise ValueError("No images loaded. Check your processed images directory.")
    print(f"Loaded {len(images)} images for search.")
except Exception as e:
    print(f"Error loading images: {e}")
    exit(1)

# -----------------------------
# Load FAISS index
# -----------------------------
try:
    index = load_faiss_index(FAISS_INDEX_FILE)

    # Ensure index is compatible for searching
    if not isinstance(index, faiss.Index):
        raise TypeError("Loaded FAISS index is not a valid FAISS index object.")

    print("FAISS index loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    exit(1)

# -----------------------------
# Perform search
# -----------------------------
while True:
    query = input("\nEnter your text query (or 'exit' to quit): ")
    if query.lower() == "exit":
        break

    top_k = 5
    try:
        # Search and normalize vectors if using cosine similarity
        results, scores = search_images(query, index, filenames, top_k=top_k)
        print(f"\nTop {top_k} results for query '{query}':")
        for i, (fname, score) in enumerate(zip(results, scores)):
            print(f"{i+1}. {fname} (score: {score:.4f})")
    except Exception as e:
        print(f"Error during search: {e}")
