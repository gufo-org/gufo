#!/usr/bin/env python3
"""Export a pinned RFC E2 token trace to the common-package CPU replay.

Uses only the standard library. Payload geometry uses the RFC's measured
aggregate constants (not a complete model adapter inventory). Output contains
prompt tokens; keep recovered traces and event logs outside the checkout.
"""

import argparse
import ast
import json
from pathlib import Path
import struct
import sys
import zipfile


def tokens(archive, name):
    data = archive.read(name)
    if data[:8] != b"\x93NUMPY\x01\x00":
        raise ValueError("expected version 1 NPY")
    length = struct.unpack_from("<H", data, 8)[0]
    header = ast.literal_eval(data[10:10 + length].decode("ascii"))
    if header["descr"] != "<i4" or header["fortran_order"] or len(header["shape"]) != 1:
        raise ValueError("expected a one-dimensional little-endian int32 trace")
    values = struct.unpack(f'<{header["shape"][0]}i', data[10 + length:])
    if any(value < 0 for value in values):
        raise ValueError("negative token")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("model", choices=("fn", "q27"))
    parser.add_argument("workload", choices=("w1", "w2", "w3", "w4"))
    args = parser.parse_args()
    base = args.results / "e2" / args.model / args.workload
    meta = json.loads((base / "requests.json").read_text())
    budget = json.loads((base / "e8.json").read_text())["ram_budget"]
    fixed, row = {"fn": (119328358, 27460), "q27": (243700000, 65536)}[args.model]
    print(fixed, row, budget, len(meta))
    with zipfile.ZipFile(base / "requests.npz") as archive:
        for i, record in enumerate(meta):
            prompt = tokens(archive, f"p{i}.npy")
            # Independent archive framing literal (Qwen im_start), as used in
            # the RFC trace analysis. No production token constants imported.
            stable = max((j for j, t in enumerate(prompt) if t == 248045), default=len(prompt))
            print(record["server_lifetime"], len(prompt), stable)
            print(*prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
