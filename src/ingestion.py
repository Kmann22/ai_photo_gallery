# src/ingestion.py
import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src folder
RAW_DIR = os.path.join(BASE_DIR, "../data/raw/")
PROCESSED_DIR = os.path.join(BASE_DIR, "../data/processed/")
IMAGE_SIZE = (224, 224)


def load_and_preprocess_images(save_processed=True):
    """
    Load all images from RAW_DIR, resize, and optionally save to PROCESSED_DIR.
    Returns:
        images: list of PIL Image objects
        filenames: list of corresponding filenames
    """
    if save_processed and not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    images = []
    filenames = []

    for filename in os.listdir(RAW_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(RAW_DIR, filename)
            try:
                img = Image.open(path).convert("RGB")
                img_resized = img.resize(IMAGE_SIZE)
                
                if save_processed:
                    save_path = os.path.join(PROCESSED_DIR, filename)
                    img_resized.save(save_path)

                images.append(img_resized)
                filenames.append(filename)

            except Exception as e:
                print(f"Error loading {filename}: {e}")

    print(f"Loaded {len(images)} images.")
    return images, filenames


# Example usage
if __name__ == "__main__":
    imgs, names = load_and_preprocess_images()
    print("Sample filenames:", names[:5])
