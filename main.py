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

# Paths
RAW_DATA_DIR = os.path.join("data", "raw")
DB_PATH = "metadata.db"

# Initialize geolocator
geolocator = Nominatim(user_agent="photo_metadata_app")

# ---------------- DATABASE ----------------
def init_db():
    """Initialize SQLite DB with geolocation column."""
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

# ---------------- INGEST ----------------
def process_file(file_path):
    filename = os.path.basename(file_path)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Skip already processed
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

def ingest_existing_files():
    """Ingest all existing files in RAW_DATA_DIR."""
    for filename in os.listdir(RAW_DATA_DIR):
        file_path = os.path.join(RAW_DATA_DIR, filename)
        if os.path.isfile(file_path):
            process_file(file_path)

# ---------------- WATCHER ----------------
class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print(f"New file detected: {event.src_path}")
            process_file(event.src_path)

def start_watcher():
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, RAW_DATA_DIR, recursive=False)
    observer.start()
    print("Watching for new files... Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    init_db()
    ingest_existing_files()  # Ingest existing images
    start_watcher()          # Start watcher for new images
