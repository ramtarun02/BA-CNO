#!/usr/bin/env bash
# Download the DeepMind MeshGraphNets CylinderFlow dataset (Apache-2.0).
# Usage: bash scripts/cylinder/download.sh data/cylinder_flow/raw [splits...]
# The train split is large (several GB); on a Mac start with: valid test
set -euo pipefail
OUT=${1:-data/cylinder_flow/raw}
shift || true
SPLITS=${@:-"train valid test"}
BASE=https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow
mkdir -p "$OUT"
curl -fL -o "$OUT/meta.json" "$BASE/meta.json"
for s in $SPLITS; do
  echo "Downloading $s.tfrecord ..."
  curl -fL -C - -o "$OUT/$s.tfrecord" "$BASE/$s.tfrecord"
done
