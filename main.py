# main.py

import os
import numpy as np
from PIL import Image
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
images, filenames = load_and_preprocess_images(save_processed=False)  # already processed
print(f"Loaded {len(images)} images for search.")

# -----------------------------
# Load FAISS index
# -----------------------------
index = load_faiss_index(FAISS_INDEX_FILE)
print("FAISS index loaded successfully.")

# -----------------------------
# Perform search
# -----------------------------
while True:
    query = input("\nEnter your text query (or 'exit' to quit): ")
    if query.lower() == "exit":
        break

    top_k = 5
    results, scores = search_images(query, index, filenames, top_k=top_k)
    print(f"\nTop {top_k} results for query '{query}':")
    for i, (fname, score) in enumerate(zip(results, scores)):
        print(f"{i+1}. {fname} (score: {score:.4f})")
