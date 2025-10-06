import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import clip
import faiss

# ------------------- CONFIGURATION -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "clip_embeddings.pt")
FAISS_STORE_FILE = os.path.join(EMBEDDINGS_DIR, "index_store.faiss")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess_clip = clip.load("ViT-B/32", device=device)

# ------------------- LOAD EMBEDDINGS -------------------
def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        return torch.load(EMBEDDINGS_FILE)
    return {}

embeddings_dict = load_embeddings()

# ------------------- FAISS HELPERS -------------------
def load_faiss_index():
    if not os.path.exists(FAISS_STORE_FILE):
        print("No FAISS index found.")
        return None, []
    data = torch.load(FAISS_STORE_FILE)
    index = faiss.deserialize_index(data["index"])
    filenames = data["filenames"]
    print(f"✅ Loaded FAISS index with {len(filenames)} entries.")
    return index, filenames

def build_and_save_faiss_index(embeddings_dict):
    filenames = list(embeddings_dict.keys())
    if not filenames:
        print("No embeddings to index.")
        return None, []
    embeddings = torch.cat([embeddings_dict[f] for f in filenames], dim=0)
    embeddings = embeddings.numpy().astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    serialized = faiss.serialize_index(index)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    torch.save({"index": serialized, "filenames": filenames}, FAISS_STORE_FILE)
    print(f"✅ Saved FAISS index with {len(filenames)} entries to {FAISS_STORE_FILE}")
    return index, filenames

# ------------------- SEARCH FUNCTION (with caching + timing) -------------------
text_embedding_cache = {}

def search_images(prompt, embeddings_dict, top_k=10):
    index, filenames = load_faiss_index()
    if index is None:
        index, filenames = build_and_save_faiss_index(embeddings_dict)
        if index is None:
            print("No embeddings found.")
            return []

    # --- CLIP text encoding (cached) ---
    clip_start = time.time()
    if prompt not in text_embedding_cache:
        text_tokens = clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            text_embedding = model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_embedding_cache[prompt] = text_embedding.cpu().numpy().astype("float32")
    query_vec = text_embedding_cache[prompt]
    clip_end = time.time()

    # --- FAISS search ---
    faiss_start = time.time()
    scores, indices = index.search(query_vec, top_k)
    faiss_end = time.time()

    results = [(filenames[i], float(scores[0][j])) for j, i in enumerate(indices[0])]

    # Log timing
    print(f"  ⏱️ CLIP encode: {clip_end - clip_start:.4f}s | FAISS search: {faiss_end - faiss_start:.4f}s")
    return results

# ------------------- LATENCY TEST -------------------
if __name__ == "__main__":
    prompts = ["beach", "mountains", "city night", "forest", "vehicles", "people"]
    total_time = 0
    top_k = 30

    print(f"🔍 Testing {len(prompts)} prompts with top-{top_k} retrieval...\n")
    for prompt in prompts:
        start_time = time.time()
        results = search_images(prompt, embeddings_dict, top_k=top_k)
        end_time = time.time()

        elapsed = end_time - start_time
        total_time += elapsed
        print(f"Prompt: '{prompt}' | Retrieved top {top_k} images in {elapsed:.4f} seconds | Results: {len(results)}\n")

    average_time = total_time / len(prompts)
    print(f"\n✅ Average total search time per prompt: {average_time:.4f} seconds")
