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
# CLIP model for location embeddings
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# -----------------------------
# Helper functions
# -----------------------------
def convert_to_degrees(value):
    """Convert GPS coordinates to decimal degrees"""
    d, m, s = value
    return d[0]/d[1] + m[0]/m[1]/60 + s[0]/s[1]/3600

def get_location(lat, lon):
    """Reverse geocode lat/lon to a human-readable address"""
    geolocator = Nominatim(user_agent="photo_metadata_app")
    try:
        location = geolocator.reverse((lat, lon), timeout=10)
        return location.address if location else "Unknown Location"
    except GeocoderTimedOut:
        return "Location lookup timed out"

def encode_date(date_str: str):
    """Encode date as [day_norm, month_norm, year_norm]"""
    try:
        dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
        day_norm = dt.day / 31.0
        month_norm = dt.month / 12.0
        year_norm = (dt.year - 2000) / 50.0  # scale years 2000-2050
        return np.array([day_norm, month_norm, year_norm])
    except:
        return np.zeros(3)

def encode_location(location: str):
    """Encode location string using CLIP text embedding"""
    with torch.no_grad():
        text = clip.tokenize([location]).to(device)
        loc_vec = model.encode_text(text)
        loc_vec = loc_vec / loc_vec.norm(dim=-1, keepdim=True)
        return loc_vec.cpu().numpy().flatten()

def create_metadata_embedding(date_str, location_name):
    """Combine date and location into a single embedding vector"""
    date_vec = encode_date(date_str)
    location_vec = encode_location(location_name)
    return np.concatenate([date_vec, location_vec])  # ~515-d vector

# -----------------------------
# Process folder
# -----------------------------
folder_path = "."  # Current folder
metadata_vectors = []
filenames = []

for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".jpeg")):
        try:
            img = Image.open(filename)
            exif_bytes = img.info.get("exif", b"")
            if not exif_bytes:
                print(f"{filename}: No EXIF metadata found.")
                continue

            exif_data = piexif.load(exif_bytes)

            # Date
            datetime_original = exif_data["Exif"].get(
                piexif.ExifIFD.DateTimeOriginal, b""
            ).decode(errors="ignore")

            # GPS → location name
            gps_ifd = exif_data.get("GPS", {})
            location_name = "Unknown Location"
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

            # Create embedding
            metadata_vector = create_metadata_embedding(datetime_original, location_name)
            metadata_vectors.append(metadata_vector)
            filenames.append(filename)

            print(f"Processed {filename}: {metadata_vector.shape}")

        except Exception as e:
            print(f"Could not process {filename}: {e}")

# -----------------------------
# Save embeddings and filenames
# -----------------------------
if metadata_vectors:
    metadata_vectors = np.stack(metadata_vectors)
    np.save("metadata_vectors.npy", metadata_vectors)
    np.save("metadata_filenames.npy", np.array(filenames))
    print(f"Saved {len(metadata_vectors)} metadata embeddings and filenames.")
else:
    print("No metadata embeddings generated.")
