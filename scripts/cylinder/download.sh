#!/usr/bin/env bash
# Download the DeepMind MeshGraphNets CylinderFlow dataset (Apache-2.0).
# Usage: bash scripts/cylinder/download.sh data/cylinder_flow/raw [splits...]
# The train split is 13.6 GB; on a Mac start with: valid test train:50
# A split written as NAME:N fetches only a byte range holding roughly the first
# N trajectories (~13.6 MB each, plus margin). The truncated last record is
# skipped by the converter, so pass --max N to convert.py.
set -euo pipefail
OUT=${1:-data/cylinder_flow/raw}
shift || true
SPLITS=${@:-"train valid test"}
BASE=https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow
BYTES_PER_TRAJ=15000000
mkdir -p "$OUT"
curl -fL -o "$OUT/meta.json" "$BASE/meta.json"
for s in $SPLITS; do
  name=${s%%:*}
  if [[ "$s" == *:* ]]; then
    n=${s##*:}
    end=$(( n * BYTES_PER_TRAJ - 1 ))
    echo "Downloading first ~$n trajectories of $name.tfrecord (bytes 0-$end) ..."
    curl -fL -r "0-$end" -o "$OUT/$name.tfrecord" "$BASE/$name.tfrecord"
  else
    echo "Downloading $name.tfrecord ..."
    curl -fL -C - -o "$OUT/$name.tfrecord" "$BASE/$name.tfrecord"
  fi
done
