# src/filtering.py

import os
import sqlite3
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
DB_FILE = os.path.join(METADATA_DIR, "metadata.sqlite")

if not os.path.exists(METADATA_DIR):
    os.makedirs(METADATA_DIR)

# -----------------------------
# EXIF Helper Functions
# -----------------------------
def get_exif_data(image_path):
    """
    Extract EXIF data from an image.
    Returns a dict with relevant fields: DateTime, GPS info, Camera
    """
    metadata = {"filename": os.path.basename(image_path),
                "datetime": None,
                "gps_lat": None,
                "gps_lon": None,
                "camera": None}
    
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "DateTime":
                    metadata["datetime"] = value
                elif tag == "Model":
                    metadata["camera"] = value
                elif tag == "GPSInfo":
                    gps_info = {}
                    for t in value:
                        sub_tag = GPSTAGS.get(t, t)
                        gps_info[sub_tag] = value[t]
                    if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
                        metadata["gps_lat"] = convert_to_degrees(gps_info["GPSLatitude"])
                        metadata["gps_lon"] = convert_to_degrees(gps_info["GPSLongitude"])
    except Exception as e:
        print(f"Error reading EXIF for {image_path}: {e}")
    return metadata


def convert_to_degrees(value):
    """
    Convert GPS coordinates to float degrees
    """
    d, m, s = value
    return float(d) + float(m)/60 + float(s)/3600


# -----------------------------
# Database Functions
# -----------------------------
def create_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            filename TEXT PRIMARY KEY,
            datetime TEXT,
            gps_lat REAL,
            gps_lon REAL,
            camera TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_metadata(metadata_list):
    """
    metadata_list: list of dicts returned by get_exif_data()
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    for meta in metadata_list:
        c.execute("""
            INSERT OR REPLACE INTO photos (filename, datetime, gps_lat, gps_lon, camera)
            VALUES (?, ?, ?, ?, ?)
        """, (meta["filename"], meta["datetime"], meta["gps_lat"], meta["gps_lon"], meta["camera"]))
    conn.commit()
    conn.close()


# -----------------------------
# Filtering Functions
# -----------------------------
def filter_by_date(start_date, end_date):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT filename FROM photos WHERE datetime BETWEEN ? AND ?", (start_date, end_date))
    results = [row[0] for row in c.fetchall()]
    conn.close()
    return results


def filter_by_location(lat_min, lat_max, lon_min, lon_max):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT filename FROM photos
        WHERE gps_lat BETWEEN ? AND ? AND gps_lon BETWEEN ? AND ?
    """, (lat_min, lat_max, lon_min, lon_max))
    results = [row[0] for row in c.fetchall()]
    conn.close()
    return results


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    from ingestion import load_and_preprocess_images
    import glob

    create_db()

    # Process all images in processed folder
    images_folder = os.path.join(BASE_DIR, "data/processed")
    image_paths = glob.glob(os.path.join(images_folder, "*.jpg"))

    metadata_list = []
    for path in image_paths:
        meta = get_exif_data(path)
        metadata_list.append(meta)

    insert_metadata(metadata_list)

    # Example filters
    print("Images between 2025:01:01 and 2025:12:31")
    print(filter_by_date("2025:01:01", "2025:12:31"))

    print("Images near lat 18.5–19.0, lon 73.7–74.0")
    print(filter_by_location(18.5, 19.0, 73.7, 74.0))
