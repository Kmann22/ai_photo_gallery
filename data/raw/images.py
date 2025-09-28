import os
from PIL import Image
import piexif

# Get the current folder
folder_path = "."

# Loop through all files in the current folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp")):
        print(f"\n=== Metadata for {filename} ===")
        try:
            img = Image.open(filename)

            # Basic image info
            print("Basic Image Info:")
            print(f"Format: {img.format}")
            print(f"Mode: {img.mode}")
            print(f"Size: {img.size}")
            print(f"Info: {img.info}")

            # EXIF metadata
            print("\nEXIF Metadata:")
            exif_data = piexif.load(img.info.get("exif", b""))
            for ifd in exif_data:
                print(f"\n{ifd} IFD:")
                for tag in exif_data[ifd]:
                    try:
                        tag_name = piexif.TAGS[ifd][tag]["name"]
                        print(f"{tag_name}: {exif_data[ifd][tag]}")
                    except KeyError:
                        print(f"Unknown Tag {tag}: {exif_data[ifd][tag]}")

        except Exception as e:
            print(f"Could not process {filename}: {e}")
