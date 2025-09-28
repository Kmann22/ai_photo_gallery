# 2nd
# src/embeddings.py

import os
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# -----------------------------
#  Paths (absolute, robust)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
CLIP_EMBED_FILE = os.path.join(EMBEDDINGS_DIR, "clip_vectors.npy")

if not os.path.exists(EMBEDDINGS_DIR):
    os.makedirs(EMBEDDINGS_DIR)

# -----------------------------
#  Load CLIP model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# -----------------------------
#  Functions
# -----------------------------
def generate_clip_embeddings(images):
    """
    images: list of PIL Images
    returns: numpy array of normalized embeddings
    """
    embeddings = []

    with torch.no_grad():
        for img in images:
            inputs = processor(images=img, return_tensors="pt").to(device)
            image_features = model.get_image_features(**inputs)
            # Normalize for cosine similarity
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(image_features.cpu().numpy())

    embeddings = np.vstack(embeddings)
    print(f"Generated embeddings for {len(embeddings)} images. Shape: {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings, file_path=CLIP_EMBED_FILE):
    np.save(file_path, embeddings)
    print(f"Saved embeddings to: {file_path}")


# -----------------------------
#  Example Usage
# -----------------------------
if __name__ == "__main__":
    from ingestion import load_and_preprocess_images

    # Step 1: Load images
    images, filenames = load_and_preprocess_images()
    print(f"Loaded {len(images)} images for embeddings.")

    # Step 2: Generate embeddings
    clip_embeddings = generate_clip_embeddings(images)

    # Step 3: Save embeddings
    save_embeddings(clip_embeddings)

    # Step 4: Test loading
    loaded = np.load(CLIP_EMBED_FILE)
    print("Loaded embeddings shape:", loaded.shape)
