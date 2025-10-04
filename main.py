import os
import sqlite3
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Paths
RAW_DATA_DIR = os.path.join("data", "raw")
DB_PATH = "metadata.db"

# Initialize geolocator
geolocator = Nominatim(user_agent="photo_metadata_app")

def init_db():
    """Initialize SQLite DB with a metadata table, including geolocation."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS photos_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            geolocation TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_exif_data(file_path):
    """Extract EXIF data from image."""
    try:
        img = Image.open(file_path)
        exif_data = img._getexif()
        if exif_data:
            return {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
    except Exception as e:
        print(f"EXIF error for {file_path}: {e}")
    return {}

def get_date(exif):
    """Get Date Taken from EXIF, fallback to None."""
    date_str = exif.get("DateTimeOriginal")
    if date_str:
        dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
        return dt.year, dt.month, dt.day
    return None, None, None

def get_modified_date(file_path):
    """Fallback: extract date from file modified time."""
    stat = os.stat(file_path)
    modified_time = datetime.fromtimestamp(stat.st_mtime)
    return modified_time.year, modified_time.month, modified_time.day

def convert_to_degrees(value):
    """Convert GPS EXIF coordinates to decimal degrees, handling IFDRational."""
    try:
        d, m, s = value
        d = float(d)
        m = float(m)
        s = float(s)
        return d + (m / 60.0) + (s / 3600.0)
    except Exception as e:
        print(f"Error converting to degrees: {e}")
        return None

def get_gps(exif):
    """Extract latitude and longitude from EXIF GPS info."""
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
    """Convert lat/lon to human-readable location."""
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

def ingest_photos():
    """Ingest photos and store metadata including geolocation in DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for filename in os.listdir(RAW_DATA_DIR):
        file_path = os.path.join(RAW_DATA_DIR, filename)
        if not os.path.isfile(file_path):
            continue

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

        # Store filename, date, and geolocation in DB
        cursor.execute("""
            INSERT INTO photos_metadata (filename, year, month, day, geolocation)
            VALUES (?, ?, ?, ?, ?)
        """, (filename, year, month, day, geolocation))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    ingest_photos()
