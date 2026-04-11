#!/usr/bin/env bash
set -euo pipefail

FILE_ID="1VQScyKlXIkVsCXMCt9Gc9g9KK8sVEKei"
DATA_DIR="$(dirname "$0")/data"
OUTPUT="$DATA_DIR/dataset.zip"

mkdir -p "$DATA_DIR"

# Prefer gdown (handles large-file confirmation automatically)
if command -v gdown &>/dev/null; then
    echo "Downloading with gdown..."
    gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$OUTPUT"
else
    echo "gdown not found, falling back to curl..."
    echo "  (install gdown for large-file support: pip install gdown)"

    CONFIRM=$(curl -sc /tmp/gdrive_cookie \
        "https://drive.google.com/uc?export=download&id=$FILE_ID" \
        | sed -n 's/.*confirm=\([0-9A-Za-z_-]*\).*/\1/p' | head -1)

    curl -Lb /tmp/gdrive_cookie \
        "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=$FILE_ID" \
        -o "$OUTPUT"
fi

echo "Saved to $OUTPUT"

# If the download is a zip/tar, uncomment the relevant line to auto-extract:
# unzip -q "$OUTPUT" -d "$DATA_DIR"
# tar -xf "$OUTPUT" -C "$DATA_DIR"
