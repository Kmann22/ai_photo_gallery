import os
import requests
from time import sleep

SAVE_DIR = 'free_images'
TOTAL_IMAGES = 500
os.makedirs(SAVE_DIR, exist_ok=True)

def download_image(url, path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {path}")
    except Exception as e:
        print(f"Failed: {path}, Error: {e}")

# Use free image placeholders from picsum.photos
for i in range(1, TOTAL_IMAGES + 1):
    img_url = f"https://picsum.photos/800/600?random={i}"  # random free image
    filename = os.path.join(SAVE_DIR, f'image_{i}.jpg')
    download_image(img_url, filename)
    sleep(0.1)  # slight delay to be polite

print(f"Finished downloading {TOTAL_IMAGES} free images.")
