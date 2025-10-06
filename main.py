import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sklearn.cluster import KMeans

import torch
import clip
import faiss
import numpy as np

# ------------------- CONFIGURATION -------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "clip_embeddings.pt")
FAISS_STORE_FILE = os.path.join(EMBEDDINGS_DIR, "index_store.faiss")
DB_PATH = os.path.join(BASE_DIR, "metadata.db")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess_clip = clip.load("ViT-B/32", device=device)
geolocator = Nominatim(user_agent="photo_metadata_app")

# ------------------- DATABASE -------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS photos_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                geolocation TEXT
            )
        """)

# ------------------- EXIF & DATE -------------------
def get_exif_data(file_path):
    try:
        img = Image.open(file_path)
        exif_data = img._getexif()
        if exif_data:
            return {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
    except Exception as e:
        print(f"EXIF error for {file_path}: {e}")
    return {}

def get_date(exif):
    date_str = exif.get("DateTimeOriginal")
    if date_str:
        dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
        return dt.year, dt.month, dt.day
    return None, None, None

def get_modified_date(file_path):
    modified_time = datetime.fromtimestamp(os.stat(file_path).st_mtime)
    return modified_time.year, modified_time.month, modified_time.day

# ------------------- GPS & GEO -------------------
def convert_to_degrees(value):
    try:
        d, m, s = value
        return float(d) + float(m)/60 + float(s)/3600
    except Exception as e:
        print(f"Error converting to degrees: {e}")
        return None

def get_gps(exif):
    gps_info = exif.get("GPSInfo")
    if not gps_info:
        return None, None

    gps_data = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}
    lat = lon = None

    try:
        if "GPSLatitude" in gps_data and "GPSLatitudeRef" in gps_data:
            lat = convert_to_degrees(gps_data["GPSLatitude"])
            if gps_data["GPSLatitudeRef"].upper() == "S":
                lat = -lat
        if "GPSLongitude" in gps_data and "GPSLongitudeRef" in gps_data:
            lon = convert_to_degrees(gps_data["GPSLongitude"])
            if gps_data["GPSLongitudeRef"].upper() == "W":
                lon = -lon
    except Exception as e:
        print(f"GPS parsing error: {e}")
    return lat, lon

def get_location(lat, lon):
    if lat is None or lon is None:
        return "N/A"
    try:
        location = geolocator.reverse((lat, lon), timeout=10)
        return location.address if location else "Unknown location"
    except GeocoderTimedOut:
        return "Geocoder timed out"
    except Exception as e:
        print(f"Geocoding error: {e}")
        return "Unknown location"

# ------------------- EMBEDDINGS -------------------
def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        return torch.load(EMBEDDINGS_FILE)
    return {}

def save_embeddings(embeddings_dict):
    torch.save(embeddings_dict, EMBEDDINGS_FILE)

def preprocess_and_store(file_path, embeddings_dict):
    filename = os.path.basename(file_path)
    processed_path = os.path.join(PROCESSED_DIR, filename)

    if filename in embeddings_dict and os.path.exists(processed_path):
        return processed_path

    try:
        img = Image.open(file_path).convert("RGB")
        img.resize((224, 224)).save(processed_path)

        image_input = preprocess_clip(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embeddings_dict[filename] = embedding.cpu()
        save_embeddings(embeddings_dict)

        print(f"Processed and stored: {filename}")
        return processed_path
    except Exception as e:
        print(f"Failed processing {file_path}: {e}")
        return None

# ------------------- FILE PROCESSING -------------------
def process_file(file_path, embeddings_dict):
    filename = os.path.basename(file_path)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM photos_metadata WHERE filename=?", (filename,))
        if cursor.fetchone():
            print(f"Skipping already processed: {filename}")
            return

    exif = get_exif_data(file_path)
    year, month, day = get_date(exif)
    if year is None:
        year, month, day = get_modified_date(file_path)
        print(f"File: {filename} | Using Modified Date -> {year}-{month}-{day}")
    else:
        print(f"File: {filename} | Using Date Taken -> {year}-{month}-{day}")

    lat, lon = get_gps(exif)
    geolocation = get_location(lat, lon)
    print(f"Geolocation: {geolocation}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO photos_metadata (filename, year, month, day, geolocation)
            VALUES (?, ?, ?, ?, ?)
        """, (filename, year, month, day, geolocation))

    preprocess_and_store(file_path, embeddings_dict)

# ------------------- INITIAL INGEST -------------------
def ingest_existing_files(embeddings_dict):
    for filename in os.listdir(RAW_DATA_DIR):
        file_path = os.path.join(RAW_DATA_DIR, filename)
        if os.path.isfile(file_path) and filename.lower().endswith((".jpg", ".jpeg", ".png")):
            process_file(file_path, embeddings_dict)

# ------------------- FAISS -------------------
def save_faiss_index(index, filenames):
    serialized = faiss.serialize_index(index)
    torch.save({"index": serialized, "filenames": filenames}, FAISS_STORE_FILE)
    print(f"✅ Saved FAISS index with {len(filenames)} entries to {FAISS_STORE_FILE}")

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
    save_faiss_index(index, filenames)
    return index, filenames

def search_images(prompt, embeddings_dict, top_k=10):
    index, filenames = load_faiss_index()
    if index is None:
        index, filenames = build_and_save_faiss_index(embeddings_dict)
        if index is None:
            print("No embeddings found.")
            return []

    text_tokens = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    query_vec = text_embedding.cpu().numpy().astype("float32")
    scores, indices = index.search(query_vec, top_k)
    return [(filenames[i], float(scores[0][j])) for j, i in enumerate(indices[0])]

# ------------------- ALBUM GENERATION -------------------
from sklearn.cluster import AgglomerativeClustering

def generate_albums(embeddings_dict, num_clusters=6):
    """
    Groups images into semantically meaningful albums using CLIP embeddings +
    hierarchical clustering and automatic album naming via CLIP similarity.
    """
    if not embeddings_dict:
        print("⚠️ No embeddings available to cluster.")
        return {}

    print("🧠 Generating smarter AI albums...")

    filenames = list(embeddings_dict.keys())
    embeddings = torch.cat([embeddings_dict[f] for f in filenames], dim=0)
    embeddings_np = embeddings.numpy().astype("float32")

    # Use hierarchical clustering (better for uneven semantic groups)
    clustering = AgglomerativeClustering(n_clusters=num_clusters)
    labels = clustering.fit_predict(embeddings_np)

    # Define conceptual album categories for naming
    concept_labels = [
        "Nature", "People", "Architecture", "Animals", "Food", "Sports",
        "Art", "Technology", "Beach", "Mountains", "Vehicles", "Night Sky"
    ]

    # Encode concept titles using CLIP for similarity-based naming
    text_tokens = clip.tokenize(concept_labels).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    albums = {}
    for cluster_id in range(num_clusters):
        cluster_imgs = [filenames[i] for i, label in enumerate(labels) if label == cluster_id]
        if not cluster_imgs:
            continue

        # Compute average embedding for cluster
        cluster_emb = torch.stack([embeddings_dict[f] for f in cluster_imgs]).mean(dim=0)
        cluster_emb = cluster_emb / cluster_emb.norm(dim=-1, keepdim=True)

        # Find the most semantically similar concept
        sims = (cluster_emb @ text_embeddings.T).squeeze()
        best_idx = sims.argmax().item()
        album_title = concept_labels[best_idx]

        # Handle duplicates (e.g., multiple "Nature" clusters)
        if album_title in albums:
            suffix = len([k for k in albums.keys() if k.startswith(album_title)]) + 1
            album_title = f"{album_title} {suffix}"

        albums[album_title] = cluster_imgs

    print(f"✅ Created {len(albums)} smarter albums with semantic grouping.")
    return albums

# ------------------- IMAGE-BASED SEARCH -------------------
def search_images_by_file(image_file, embeddings_dict, top_k=10):
    """
    Given a file-like object (from Flask upload), return top_k similar images.
    """
    # Load FAISS index
    index, filenames = load_faiss_index()
    if index is None:
        index, filenames = build_and_save_faiss_index(embeddings_dict)
        if index is None:
            print("No embeddings found.")
            return []

    try:
        # Open and preprocess image
        img = Image.open(image_file).convert("RGB")
        image_input = preprocess_clip(img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        query_vec = embedding.cpu().numpy().astype("float32")
        scores, indices = index.search(query_vec, top_k)
        return [(filenames[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
    except Exception as e:
        print(f"Image search failed: {e}")
        return []

def get_filter_options():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT year FROM photos_metadata WHERE year IS NOT NULL ORDER BY year DESC")
        years = [row[0] for row in cursor.fetchall()]

        cursor.execute("SELECT DISTINCT geolocation FROM photos_metadata WHERE geolocation NOT IN ('N/A','Unknown location') ORDER BY geolocation")
        locations = [row[0] for row in cursor.fetchall()]

    return years, locations

# ------------------- FLASK APP -------------------
app = Flask(__name__)

# Load embeddings globally
embeddings_dict = load_embeddings()
albums_cache = generate_albums(embeddings_dict, num_clusters=5)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/search", methods=["POST"])
def search_route():
    query = request.form.get("query")
    if not query:
        return render_template("home.html", error="Please enter a search query.")

    results = search_images(query, embeddings_dict, top_k=30)
    image_paths = [f"/processed/{fname}" for fname, _ in results if os.path.exists(os.path.join(PROCESSED_DIR, fname))]
    return render_template("gallery.html", query=query, image_paths=image_paths)

@app.route("/search_image", methods=["POST"])
def search_image_route():
    image_file = request.files.get("image")
    if not image_file:
        return render_template("home.html", error="Please upload an image.")

    results = search_images_by_file(image_file, embeddings_dict, top_k=30)
    image_paths = [f"/processed/{fname}" for fname, _ in results if os.path.exists(os.path.join(PROCESSED_DIR, fname))]
    return render_template("gallery.html", query="Image Search", image_paths=image_paths)

@app.route("/processed/<path:filename>")
def serve_processed(filename):
    return send_from_directory(PROCESSED_DIR, filename)

@app.route("/albums")
def view_albums():
    return render_template("album.html", albums=albums_cache)

@app.route("/album/<album_name>")
def view_single_album(album_name):
    album_name = album_name.replace("%20", " ")
    if album_name not in albums_cache:
        return f"Album '{album_name}' not found.", 404
    image_paths = [f"/processed/{fname}" for fname in albums_cache[album_name]]
    return render_template("gallery.html", query=album_name, image_paths=image_paths)

@app.route("/filter", methods=["GET", "POST"])
def filter_photos():
    years, locations = get_filter_options()

    if request.method == "POST":
        selected_year = request.form.get("year")
        selected_location = request.form.get("location")

        query = "SELECT filename FROM photos_metadata WHERE 1=1"
        params = []

        if selected_year:
            query += " AND year=?"
            params.append(selected_year)

        if selected_location:
            query += " AND geolocation LIKE ?"
            params.append(f"%{selected_location}%")

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        image_paths = [f"/processed/{row[0]}" for row in rows if os.path.exists(os.path.join(PROCESSED_DIR, row[0]))]

        if not image_paths:
            return render_template("filtered_gallery.html", error="No photos found.", image_paths=[])

        return render_template("filtered_gallery.html", image_paths=image_paths)

    # GET request: show filter form
    return render_template("filter.html", years=years, locations=locations)


@app.route("/dashboard")
def dashboard():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM photos_metadata")
        total_photos = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT year) FROM photos_metadata WHERE year IS NOT NULL")
        unique_years = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT geolocation) FROM photos_metadata WHERE geolocation NOT IN ('N/A','Unknown location')")
        locations_count = cursor.fetchone()[0]

        cursor.execute("SELECT filename, printf('%04d-%02d-%02d', year, month, day) AS date, geolocation FROM photos_metadata ORDER BY id DESC LIMIT 10")
        recent_photos = cursor.fetchall()

    return render_template(
        "dashboard.html",
        total_photos=total_photos,
        unique_years=unique_years,
        locations_count=locations_count,
        recent_photos=recent_photos
    )

# ------------------- MAIN -------------------
if __name__ == "__main__":
    init_db()
    ingest_existing_files(embeddings_dict)
    build_and_save_faiss_index(embeddings_dict)
    app.run(debug=True)
