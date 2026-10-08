#!/usr/bin/env python3
"""Disk cost of the cache's file shapes on this machine (NVMe under dm-crypt).

- write + fsync one file per size: chunk (56 MB, 128 MiB), fixed state
  (114, 232 MiB), full snapshots (1, 4 GiB)
- cold read of the same files (page cache dropped with POSIX_FADV_DONTNEED)
- restore shapes: one 4 GiB file versus 72 chunk files of 56 MiB (same bytes)
"""
import json
import os
import pathlib
import time

from paths import WORK as HERE
DIR = HERE / "cache" / "diskbench"
BLOCK = 8 << 20
PAYLOAD = os.urandom(BLOCK)


def write_file(path, size):
    started = time.perf_counter()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        left = size
        while left:
            n = min(left, BLOCK)
            os.write(fd, PAYLOAD[:n])
            left -= n
        os.fsync(fd)
    finally:
        os.close(fd)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    os.fsync(dir_fd)
    os.close(dir_fd)
    return time.perf_counter() - started


def drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    os.fdatasync(fd) if False else None
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def read_file(path):
    started = time.perf_counter()
    fd = os.open(path, os.O_RDONLY)
    try:
        while os.read(fd, BLOCK):
            pass
    finally:
        os.close(fd)
    return time.perf_counter() - started


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    sizes = {"chunk_56MiB": 56 << 20, "fixed_114MiB": 114 << 20,
             "chunk27b_128MiB": 128 << 20, "fixed27b_232MiB": 232 << 20,
             "full_1GiB": 1 << 30, "full_4GiB": 4 << 30}
    for name, size in sizes.items():
        path = DIR / f"{name}.bin"
        writes = [write_file(path, size) for _ in range(3)]
        drop_cache(path)
        cold = read_file(path)
        rows.append({"file": name, "bytes": size,
                     "write_fsync_s": round(min(writes), 4),
                     "write_gb_per_s": round(size / min(writes) / 1e9, 3),
                     "cold_read_s": round(cold, 4),
                     "cold_read_gb_per_s": round(size / cold / 1e9, 3)})
        print(json.dumps(rows[-1]), flush=True)
        path.unlink()
    # Restore shapes with the same bytes: one file versus 72 chunk files.
    chunk = 56 << 20
    paths = [DIR / f"c{i:03d}.bin" for i in range(72)]
    started = time.perf_counter()
    for path in paths:
        write_file(path, chunk)
    write_many = time.perf_counter() - started
    for path in paths:
        drop_cache(path)
    started = time.perf_counter()
    for path in paths:
        read_file(path)
    read_many = time.perf_counter() - started
    big = DIR / "one.bin"
    write_one = write_file(big, chunk * 72)
    drop_cache(big)
    read_one = read_file(big)
    shapes = {"bytes": chunk * 72, "write_72_chunks_s": round(write_many, 3),
              "write_one_file_s": round(write_one, 3),
              "cold_read_72_chunks_s": round(read_many, 3),
              "cold_read_one_file_s": round(read_one, 3)}
    print(json.dumps(shapes), flush=True)
    for path in paths + [big]:
        path.unlink()
    (HERE / "results" / "diskbench.json").write_text(
        json.dumps({"files": rows, "restore_shapes": shapes}, indent=1))


if __name__ == "__main__":
    main()
