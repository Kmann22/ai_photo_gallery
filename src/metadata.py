import os
from PIL import Image
import piexif
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import numpy as np
import clip
import torch
from datetime import datetime

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
METADATA_EMBED_FILE = os.path.join(EMBEDDINGS_DIR, "metadata_vectors.npy")
METADATA_FILENAMES_FILE = os.path.join(EMBEDDINGS_DIR, "metadata_filenames.npy")

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# -----------------------------
# CLIP model for location embeddings
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)
model.eval()

# -----------------------------
# Helper functions
# -----------------------------
def convert_to_degrees(value):
    d, m, s = value
    return d[0]/d[1] + m[0]/m[1]/60 + s[0]/s[1]/3600

def get_location(lat, lon):
    geolocator = Nominatim(user_agent="photo_metadata_app")
    try:
        location = geolocator.reverse((lat, lon), timeout=10)
        return location.address if location else ""
    except GeocoderTimedOut:
        return ""

def encode_date(date_str: str):
    """Encode date as [day_norm, month_norm, year_norm]"""
    try:
        dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
        return np.array([dt.day/31.0, dt.month/12.0, (dt.year-2000)/50.0])
    except:
        return np.zeros(3)

def encode_location(location: str):
    """Encode location using CLIP text encoder"""
    if not location:
        return np.zeros(512)  # blank if location missing
    with torch.no_grad():
        text = clip.tokenize([location]).to(device)
        loc_vec = model.encode_text(text)
        loc_vec = loc_vec / loc_vec.norm(dim=-1, keepdim=True)
        return loc_vec.cpu().numpy().flatten()

def create_metadata_embedding(exif_data):
    # Date
    date_taken = exif_data.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal, b"").decode(errors="ignore")
    date_vec = encode_date(date_taken)

    # GPS → location
    gps_ifd = exif_data.get("GPS", {})
    location_name = ""
    if gps_ifd:
        lat = gps_ifd.get(piexif.GPSIFD.GPSLatitude)
        lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef)
        lon = gps_ifd.get(piexif.GPSIFD.GPSLongitude)
        lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef)
        if lat and lon and lat_ref and lon_ref:
            lat_deg = convert_to_degrees(lat)
            if lat_ref == b"S":
                lat_deg = -lat_deg
            lon_deg = convert_to_degrees(lon)
            if lon_ref == b"W":
                lon_deg = -lon_deg
            location_name = get_location(lat_deg, lon_deg)

    location_vec = encode_location(location_name)
    return np.concatenate([date_vec, location_vec])  # 515-d vector

# -----------------------------
# Main function
# -----------------------------
def generate_metadata_embeddings():
    embeddings = []
    filenames = []

    for filename in os.listdir(RAW_DIR):
        if filename.lower().endswith((".jpg", ".jpeg")):
            img_path = os.path.join(RAW_DIR, filename)
            try:
                img = Image.open(img_path)
                exif_bytes = img.info.get("exif", b"")
                exif_data = piexif.load(exif_bytes) if exif_bytes else {}
                emb = create_metadata_embedding(exif_data)
                embeddings.append(emb)
                filenames.append(filename)
                print(f"Processed {filename}")
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    if embeddings:
        embeddings = np.stack(embeddings)
        np.save(METADATA_EMBED_FILE, embeddings)
        np.save(METADATA_FILENAMES_FILE, np.array(filenames))
        print(f"Saved {len(embeddings)} metadata embeddings to {METADATA_EMBED_FILE}")
    else:
        print("No metadata embeddings generated.")

    return embeddings, filenames

# -----------------------------
# Run if main
# -----------------------------
if __name__ == "__main__":
    generate_metadata_embeddings()
