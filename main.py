import os
import sqlite3
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import torch
import clip

# ---------------- PROJECT-ROOT SAFE PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "clip_embeddings.pt")
DB_PATH = os.path.join(BASE_DIR, "metadata.db")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ---------------- CLIP SETUP ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess_clip = clip.load("ViT-B/32", device=device)

# ---------------- GEO LOCATOR ----------------
geolocator = Nominatim(user_agent="photo_metadata_app")

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS photos_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            geolocation TEXT
        )
    """)
    conn.commit()
    conn.close()

# ---------------- EXIF & DATE ----------------
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
    stat = os.stat(file_path)
    modified_time = datetime.fromtimestamp(stat.st_mtime)
    return modified_time.year, modified_time.month, modified_time.day

# ---------------- GPS & GEO ----------------
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
            if gps_data["GPSLatitudeRef"] in ["S", "s"]:
                lat = -lat
        if "GPSLongitude" in gps_data and "GPSLongitudeRef" in gps_data:
            lon = convert_to_degrees(gps_data["GPSLongitude"])
            if gps_data["GPSLongitudeRef"] in ["W", "w"]:
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

# ---------------- EMBEDDINGS ----------------
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
        img_resized = img.resize((224, 224))
        img_resized.save(processed_path)

        # CLIP embedding
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

# ---------------- FILE PROCESSING ----------------
def process_file(file_path, embeddings_dict):
    filename = os.path.basename(file_path)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM photos_metadata WHERE filename=?", (filename,))
    if cursor.fetchone():
        print(f"Skipping already processed: {filename}")
        conn.close()
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

    cursor.execute("""
        INSERT INTO photos_metadata (filename, year, month, day, geolocation)
        VALUES (?, ?, ?, ?, ?)
    """, (filename, year, month, day, geolocation))
    conn.commit()
    conn.close()

    preprocess_and_store(file_path, embeddings_dict)

# ---------------- INITIAL INGEST ----------------
def ingest_existing_files(embeddings_dict):
    for filename in os.listdir(RAW_DATA_DIR):
        file_path = os.path.join(RAW_DATA_DIR, filename)
        if os.path.isfile(file_path) and filename.lower().endswith((".jpg", ".jpeg", ".png")):
            process_file(file_path, embeddings_dict)

# ---------------- WATCHER ----------------
class NewFileHandler(FileSystemEventHandler):
    def __init__(self, embeddings_dict):
        self.embeddings_dict = embeddings_dict

    def on_created(self, event):
        if not event.is_directory:
            print(f"New file detected: {event.src_path}")
            process_file(event.src_path, self.embeddings_dict)

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

# ---------------- MAIN ----------------
if __name__ == "__main__":
    init_db()
    embeddings_dict = load_embeddings()
    ingest_existing_files(embeddings_dict)
    start_watcher(embeddings_dict)
