import os
import time
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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

# ------------------- WATCHER -------------------
class NewFileHandler(FileSystemEventHandler):
    def __init__(self, embeddings_dict):
        self.embeddings_dict = embeddings_dict

    def on_created(self, event):
        if not event.is_directory:
            print(f"New file detected: {event.src_path}")
            process_file(event.src_path, self.embeddings_dict)
            build_and_save_faiss_index(self.embeddings_dict)

def start_watcher(embeddings_dict):
    event_handler = NewFileHandler(embeddings_dict)
    observer = Observer()
    observer.schedule(event_handler, RAW_DATA_DIR, recursive=False)
    observer.start()
    print("Watching for new files...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

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

# ------------------- FLASK FRONTEND -------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    if not query:
        return render_template("home.html", error="Please enter a search query.")

    results = search_images(query, embeddings_dict, top_k=30)
    image_paths = [f"/processed/{fname}" for fname, _ in results if os.path.exists(os.path.join(PROCESSED_DIR, fname))]

    return render_template("gallery.html", query=query, image_paths=image_paths)

# Serve images directly from data/processed without static/
@app.route("/processed/<path:filename>")
def serve_processed(filename):
    return send_from_directory(PROCESSED_DIR, filename)

# ------------------- MAIN -------------------
if __name__ == "__main__":
    init_db()
    embeddings_dict = load_embeddings()
    ingest_existing_files(embeddings_dict)
    build_and_save_faiss_index(embeddings_dict)
    app.run(debug=True)
