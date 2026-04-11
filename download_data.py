#!/usr/bin/env python3
"""Download dataset from Google Drive."""

import os
import gdown

FILE_ID = "1VQScyKlXIkVsCXMCt9Gc9g9KK8sVEKei"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT = os.path.join(DATA_DIR, "crypto_data.csv")

os.makedirs(DATA_DIR, exist_ok=True)

url = f"https://drive.google.com/uc?id={FILE_ID}"
print(f"Downloading to {OUTPUT} ...")
gdown.download(url, OUTPUT, quiet=False)
print("Done.")

# To auto-extract after download, uncomment:
# import zipfile
# with zipfile.ZipFile(OUTPUT, "r") as z:
#     z.extractall(DATA_DIR)
