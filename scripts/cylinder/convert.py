"""Convert DeepMind CylinderFlow TFRecords to per-trajectory .npz (no TensorFlow needed).

    bash scripts/cylinder/download.sh data/cylinder_flow/raw
    python scripts/cylinder/convert.py --raw data/cylinder_flow/raw --out data/cylinder_flow --splits valid test
    python scripts/cylinder/convert.py --raw data/cylinder_flow/raw --out data/cylinder_flow --splits train --max 50   # Mac prototyping
"""

from __future__ import annotations

import argparse

from ba_cno.data.tfrecord import convert_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--max", type=int, default=None, help="max trajectories per split")
    args = ap.parse_args()
    for split in args.splits:
        n = convert_split(args.raw, split, args.out, args.max)
        print(f"{split}: {n} trajectories -> {args.out}/{split}")


if __name__ == "__main__":
    main()
