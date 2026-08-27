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
  echo "Unzipping (records100/ only -- skipping records500/ to save disk)..."
  TOP_DIR=$(unzip -Z1 "$ZIP_PATH" | head -n 1 | cut -d/ -f1) || true
  unzip -q -o "$ZIP_PATH" -d "$DATA_DIR" \
    "$TOP_DIR/records100/*" \
    "$TOP_DIR/*.csv" \
    "$TOP_DIR/LICENSE.txt" \
    "$TOP_DIR/SHA256SUMS.txt"
  mv "$DATA_DIR/$TOP_DIR" "$DATA_DIR/ptbxl"
  rm "$ZIP_PATH"
else
  echo "data/ptbxl already exists, skipping download."
fi

echo "Done. records100/ present: $([ -d "$DATA_DIR/ptbxl/records100" ] && echo yes || echo no)"
