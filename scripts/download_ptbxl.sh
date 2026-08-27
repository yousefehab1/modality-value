#!/usr/bin/env bash
# PTB-XL, PhysioNet, open access, no credentialing application required.
# CC BY 4.0. https://physionet.org/content/ptb-xl/1.0.3/
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"
ZIP_URL="https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
ZIP_PATH="$DATA_DIR/ptbxl.zip"

mkdir -p "$DATA_DIR"

EXPECTED_ZIP_SIZE=1839504686

if [ ! -d "$DATA_DIR/ptbxl" ]; then
  if [ ! -f "$ZIP_PATH" ] || [ "$(stat -f%z "$ZIP_PATH" 2>/dev/null || stat -c%s "$ZIP_PATH")" -lt "$EXPECTED_ZIP_SIZE" ]; then
    echo "Downloading PTB-XL (this is ~1.7GB, may take a while; resumable)..."
    curl -C - -L -o "$ZIP_PATH" "$ZIP_URL"
  else
    echo "Zip already fully downloaded, skipping curl."
  fi
  echo "Unzipping..."
  unzip -q "$ZIP_PATH" -d "$DATA_DIR"
  EXTRACTED_DIR=$(find "$DATA_DIR" -maxdepth 1 -type d -iname "ptb-xl-*" | head -n 1)
  mv "$EXTRACTED_DIR" "$DATA_DIR/ptbxl"
  rm "$ZIP_PATH"
else
  echo "data/ptbxl already exists, skipping download."
fi

# Keep the 100Hz set (tractable on 16GB), drop the 500Hz set.
if [ -d "$DATA_DIR/ptbxl/records500" ]; then
  echo "Deleting records500/ (100Hz set only)..."
  rm -rf "$DATA_DIR/ptbxl/records500"
fi

echo "Done. records100/ present: $([ -d "$DATA_DIR/ptbxl/records100" ] && echo yes || echo no)"
