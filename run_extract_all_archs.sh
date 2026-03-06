#!/bin/bash
# Chạy extract_features.py cho tất cả các kiến trúc RegNet
# Output: feature_output/rny002_gsf/, feature_output/rny004_gsf/, ...

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEATURE_OUTPUT="${1:-feature_output}"
FRAME_DIR="${2:-data/soccernetball/224p}"
STORE_DIR="${3:-$FRAME_DIR}"
DATASET_PATH="${4:-data}"

mkdir -p "$FEATURE_OUTPUT"

for arch in rny002_gsf rny004_gsf rny006_gsf rny008_gsf; do
    echo "=== Extracting features with $arch ==="
    python "$SCRIPT_DIR/extract_features.py" \
        --frame-dir "$FRAME_DIR" \
        --store-dir "$STORE_DIR" \
        --feature-dir "$FEATURE_OUTPUT/$arch" \
        --dataset-path "$DATASET_PATH" \
        --feature-arch "$arch" \
        --build-clips
done

echo "Done. Features saved under: $FEATURE_OUTPUT/"
ls -la "$FEATURE_OUTPUT"/
