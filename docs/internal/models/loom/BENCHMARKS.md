# Loom package benchmarks

Throughput through Gufo, measured with `src/models/loom/tools/validation/bench.py`
(greedy, thinking off, two repetitions): decode is one stream after a short
prompt, prefill is the time to first token of a ~10k-token document (hrx-system's
`loom/docs/src/serving/http-profiles.md`), and 4 streams is aggregate decode.

## Linux

Every package QUALITY.md lists, on Strix Halo (gfx1151), Linux 7.0, from the
extension's release 0.3.6, with Gufo ec6d3d07 and Loom 0.2.0: see
[QUALITY.md](QUALITY.md), which lists each package's decode, prefill and
4-stream throughput next to its conformance result. Against loom-serve 0.2.0
on the same machine, Gufo decoded at 99-105% of its speed (the extension's
`validation/results/bench.md`).

## Windows

Strix Halo (Ryzen AI Max+ 395, gfx1151), Windows 11 Pro 10.0.26300, 96 GB
Variable Graphics Memory; Gufo `feat/loom-in-core` built with TheRock 10.0.0
clang-cl, the Loom runtime from hrx-system v0.2.0; `gufo serve llm --sessions 4`
with the package and its GGUF downloaded separately (`--loom-checkpoint`);
2026-11-21.

| Package | Decode tok/s | Prefill tok/s (10,594 tokens) | 4 streams tok/s | Linux (QUALITY.md) |
| --- | --- | --- | --- | --- |
| `loom-fun/unsloth-Qwen3-0.6B-loom` | 305.9 | 7,462 | 1,010.3 | 344.4 / 8,674 / 1,141.2 |

`gufo bench -p 2048 -n 128 -r 3` on the same build: pp2048 18,386 tok/s,
tg128 215.9 tok/s.

Windows runs 86-89% of the Linux figures; the two hosts differ in OS, memory
configuration and driver, so this is not a controlled comparison. On Windows,
libamdf submits through WDDM kernel queues (Linux uses KFD user queues).
