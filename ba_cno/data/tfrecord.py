"""Dependency-free reader for the DeepMind MeshGraphNets TFRecord datasets.

Avoids installing TensorFlow just to convert data. Implements the TFRecord
container (length, crc, payload, crc) and the small part of the
``tf.train.Example`` protobuf needed here (bytes / float / int64 lists).
The decoding of each feature follows ``meshgraphnets/dataset.py``: raw bytes are
decoded with the dtype and shape from ``meta.json``; ``static`` fields are
stored once (not tiled over time) to save disk.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterator

import numpy as np


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _iter_fields(buf: bytes) -> Iterator[tuple[int, int, object]]:
    """Yield (field_number, wire_type, value) for a serialized protobuf message."""

    pos, end = 0, len(buf)
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _read_varint(buf, pos)
        elif wire == 1:
            value, pos = buf[pos:pos + 8], pos + 8
        elif wire == 2:
            length, pos = _read_varint(buf, pos)
            value, pos = buf[pos:pos + length], pos + length
        elif wire == 5:
            value, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield field, wire, value


def _parse_feature(buf: bytes):
    for field, _, value in _iter_fields(buf):
        if field == 1:  # BytesList
            return [v for f, _, v in _iter_fields(value) if f == 1]
        if field == 2:  # FloatList (packed)
            vals = [v for f, _, v in _iter_fields(value) if f == 1]
            return np.frombuffer(b"".join(vals), dtype="<f4")
        if field == 3:  # Int64List (packed varints)
            out = []
            for f, wire, v in _iter_fields(value):
                if f == 1 and wire == 2:
                    p = 0
                    while p < len(v):
                        x, p = _read_varint(v, p)
                        out.append(x)
                elif f == 1:
                    out.append(v)
            return np.asarray(out, dtype=np.int64)
    return []


def parse_example(buf: bytes) -> dict[str, object]:
    """tf.train.Example -> {name: list[bytes] | np.ndarray}."""

    features: dict[str, object] = {}
    for field, _, features_msg in _iter_fields(buf):
        if field != 1:
            continue
        for f, _, entry in _iter_fields(features_msg):  # map<string, Feature> entries
            if f != 1:
                continue
            key, value = None, None
            for ef, _, ev in _iter_fields(entry):
                if ef == 1:
                    key = ev.decode()
                elif ef == 2:
                    value = _parse_feature(ev)
            features[key] = value
    return features


def iter_tfrecord(path: str | Path) -> Iterator[bytes]:
    """Yield record payloads. A truncated final record (partial download) is skipped."""

    with open(path, "rb") as fh:
        while True:
            header = fh.read(8)
            if len(header) < 8:
                return
            (length,) = struct.unpack("<Q", header)
            fh.read(4)  # length crc
            payload = fh.read(length)
            if len(payload) < length:
                return
            fh.read(4)  # payload crc
            yield payload


def decode_trajectory(example: dict[str, object], meta: dict) -> dict[str, np.ndarray]:
    out = {}
    for key, field in meta["features"].items():
        raw = example[key]
        raw = raw[0] if isinstance(raw, list) else raw
        arr = np.frombuffer(raw, dtype=np.dtype(field["dtype"]))
        arr = arr.reshape(field["shape"])
        if field["type"] == "static":
            arr = arr[0]
        elif field["type"] != "dynamic":
            raise ValueError(f"Unsupported field type {field['type']!r} for {key}")
        out[key] = np.ascontiguousarray(arr)
    return out


def convert_split(data_dir: str | Path, split: str, out_dir: str | Path, max_trajectories: int | None = None) -> int:
    """Convert ``<data_dir>/<split>.tfrecord`` into one ``.npz`` per trajectory."""

    data_dir, out_dir = Path(data_dir), Path(out_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data_dir / "meta.json").read_text())
    count = 0
    for idx, payload in enumerate(iter_tfrecord(data_dir / f"{split}.tfrecord")):
        if max_trajectories is not None and idx >= max_trajectories:
            break
        traj = decode_trajectory(parse_example(payload), meta)
        np.savez(out_dir / f"traj_{idx:04d}.npz", dt=np.float32(meta.get("dt", 0.01)), **traj)
        count += 1
    return count
