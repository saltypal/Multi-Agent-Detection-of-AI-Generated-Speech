#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# download_dataset.sh
# Downloads the ASVspoof5-FLAC dataset from Kaggle.
#
# Prerequisites:
#   pip install kaggle
#   export KAGGLE_USERNAME=<your_username>
#   export KAGGLE_KEY=<your_api_key>
#   OR place ~/.kaggle/kaggle.json with your credentials.
#
# Usage:
#   bash download_dataset.sh [output_dir]
#   Default output_dir: ./data/asvspoof5
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR="${1:-./data/asvspoof5}"

echo "=== ASVspoof5 Dataset Downloader ==="
echo "Output directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

curl -L -o "$OUTPUT_DIR/asvspoof5-flac.zip" \
  "https://www.kaggle.com/api/v1/datasets/download/aniket202411001/asvspoof5-flac"

echo "Unzipping..."
unzip -q "$OUTPUT_DIR/asvspoof5-flac.zip" -d "$OUTPUT_DIR"
rm "$OUTPUT_DIR/asvspoof5-flac.zip"

echo "Done! Dataset at: $OUTPUT_DIR"
ls "$OUTPUT_DIR"
