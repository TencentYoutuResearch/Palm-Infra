#!/usr/bin/env python3
"""Read-only audit of lossless BF16-max / 7 encoding for package W4G32 scales."""
import argparse
import json
import mmap
from pathlib import Path
import struct
from collections import defaultdict

import numpy as np


def encode_scales(scales):
    scales = np.asarray(scales, dtype=np.float32)
    # Ragged, padded rows may carry canonical positive-zero scales.
    if not np.all(np.isfinite(scales) & (scales >= 0)):
        return None
    maximum = (scales * np.float32(7)).view(np.uint32)
    code = ((maximum + np.uint32(32767) + ((maximum >> 16) & 1)) >> 16).astype(np.uint16)
    restored = (code.astype(np.uint32) << 16).view(np.float32) / np.float32(7)
    return code if np.array_equal(restored.view(np.uint32), scales.view(np.uint32)) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--by-shape", action="store_true",
                        help="report validated compression coverage by matrix geometry")
    args = parser.parse_args()
    unique = set()
    checked = blocks = weights = rejected = 0
    by_shape = defaultdict(lambda: dict(weights=0, encodable_weights=0,
                                       canonical_bytes=0, validated_saved_bytes=0))
    validated_saved = 0
    with args.package.open("rb") as source:
        header = struct.unpack("<II15Q", source.read(128))
        source.seek(header[2])
        metadata = json.loads(source.read(header[3]))
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as package:
            for ref, (offset, size) in metadata["weights"].items():
                start = header[12] + offset
                weight = struct.unpack("<4I4Q4Q2I", package[start:start + 88])
                if weight[3] != 3 or not weight[1] & 4:
                    continue
                data_offset, data_size = weight[8:10]
                if data_size % 160 or data_offset > size or data_size > size - data_offset:
                    raise ValueError(f"invalid W4G32 payload: {ref}")
                count = data_size // 160
                scales = np.ndarray((count, 8), dtype="<f4", buffer=package,
                    offset=start + data_offset, strides=(160, 4)).copy()
                encoded = encode_scales(scales)
                if encoded is None:
                    rejected += 1
                else:
                    validated_saved += count * 16
                if args.by_shape:
                    shape = 'x'.join(map(str, weight[4:8]))
                    record = by_shape[shape]
                    record['weights'] += 1
                    record['encodable_weights'] += int(encoded is not None)
                    record['canonical_bytes'] += count * 160
                    if encoded is not None:
                        record['validated_saved_bytes'] += count * 16
                unique.update(map(int, np.unique(scales.view(np.uint32))))
                checked += scales.size
                blocks += count
                weights += 1
    result = dict(weights=weights, scale_count=checked,
        unique_scale_bits=len(unique), nonencodable_weights=rejected,
        canonical_bytes=blocks * 160, candidate_bytes=blocks * 144,
        potential_saved_bytes=blocks * 16, validated_saved_bytes=validated_saved)
    if args.by_shape:
        result['by_shape'] = dict(sorted(by_shape.items(),
            key=lambda item: item[1]['validated_saved_bytes'], reverse=True))
    print(json.dumps(result, indent=2))
    if rejected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
