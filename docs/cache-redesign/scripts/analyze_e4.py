#!/usr/bin/env python3
"""E4: cost of the current cache on the request path, from server logs.

- live checkpoint captures: bytes and capture_ms (fit ms per GB)
- RAM refusals (snapshot skipped byte_capacity) and disk skips by reason
- disk bytes written and write_ms
"""
import json
import pathlib
import re
import sys

from paths import WORK as HERE
LIVE = re.compile(r"live_checkpoint tokens=(\d+) bytes=(\d+) capture_ms=([\d.]+)")
STORED = re.compile(r"disk_cache action=stored .*?file_bytes=(\d+) .*?tokens=(\d+)"
                    r".*?write_ms=([\d.]+)")
SKIP = re.compile(r"disk_cache action=skipped reason=(\w+)")
RAM_SKIP = re.compile(r"event=snapshot action=skipped reason=(\w+)")


def summarize(log):
    text = log.read_text(errors="replace")
    live = [(int(t), int(b), float(ms)) for t, b, ms in LIVE.findall(text)]
    stored = [(int(b), int(t), float(ms)) for b, t, ms in STORED.findall(text)]
    skips, ram_skips = {}, {}
    for reason in SKIP.findall(text):
        skips[reason] = skips.get(reason, 0) + 1
    for reason in RAM_SKIP.findall(text):
        ram_skips[reason] = ram_skips.get(reason, 0) + 1
    big = [(b, ms) for _, b, ms in live if b > 5e8]
    ms_per_gb = (sum(ms for _, ms in big) / (sum(b for b, _ in big) / 1e9)) if big else None
    return {
        "captures": len(live),
        "capture_ms_total": round(sum(ms for *_, ms in live), 1),
        "capture_ms_max": max((ms for *_, ms in live), default=0),
        "capture_ms_per_gb": round(ms_per_gb, 1) if ms_per_gb else None,
        "deepest_capture": max(live, default=None),
        "disk_writes": len(stored),
        "disk_written_gb": round(sum(b for b, *_ in stored) / 1e9, 2),
        "disk_write_ms_total": round(sum(ms for *_, ms in stored), 1),
        "disk_skips": skips,
        "ram_skips": ram_skips,
    }


def main():
    rows = {}
    for log in sorted((HERE / "results" / "e2").glob("*/*/server.log")):
        key = f"{log.parent.parent.name}/{log.parent.name}"
        rows[key] = summarize(log)
        print(key, json.dumps(rows[key]))
    (HERE / "results" / "e2" / "e4.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    sys.exit(main())
