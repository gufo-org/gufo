# SHQ review evidence, 2026-09-13

Companion to [the review](../../docs/SHQ_INVESTIGATION_REVIEW.md) of investigation revision `620f4ebf8fd6c516d8412c0bca92a7baf456796a`.

## New checks

Run from the repository root, using the project's Nix environment:

```sh
nix develop -c python3 benchmarks/shq-review-2026-09-13/paired_stats.py
nix develop -c python3 benchmarks/shq-review-2026-09-13/format_checks.py
nix develop -c python3 benchmarks/shq-review-2026-09-13/audit.py
nix develop -c python3 benchmarks/shq-review-2026-09-13/check_headers.py
```

- `paired_stats.py`: no models/network required. Recovers equal-sized chunk means by differencing the cumulative means printed in the original 24-chunk Q5/Q6 logs. Resamples **paired chunks**, 100,000 replicates, seed 731. Logged means are rounded; the interval describes this WikiText slice, not deployment generalization. It does not assume individual tokens are independent. Output retained in `paired-output.txt`.
- `format_checks.py`: small Python checks of actual SHQ6 serialized plane lengths and the **proposed**, not implemented, SHQ5 nibble/bitplane packing. Checks all signed five-bit values and an example showing a floating scale search cannot be encoded as the proposed power-of-two exponent. It is not a GPU/kernel/model qualification test.
- `audit.py`: opens local GGUFs through the in-tree reader and performs inventory arithmetic; it does not dequantize or execute the production models. Requires the named `/persist/models` artifacts. `audit-output.txt` is the actual output on this host. The broad local file discovery is a convenience: compare the printed paths against the recorded target/support and Qwen snapshot before interpreting results. “One-pass” charges one embedding/lookup row, six of 256 experts and no Qwen MTP block; it does **not** account for conditional indexer execution, KV, scratch, caching, speculative work or actual DRAM transactions. The Qwen mixed recipe is an optimistic all-matrix construction, not a qualified sensitive-tensor manifest. The final two `dense resident illustrative` lines use a separate size-threshold heuristic and are **not** the explicit scenario-table manifest.
- `check_headers.py`: network required. Reads the pinned official 0731 index and only safetensors headers via HTTP ranges, not full shards. Output retained in `headers-output.txt`. Range requests have separate cache keys and are checked for partial-content responses. The pinned index's total is tensor payload metadata, not a whole-repository download size.

No new GPU timing, DS4/Qwen quality inference, model conversion or NPU run was performed. Nix emitted its expected dirty-tree warning because these evidence files were being added. The canonical compiled PR checks are not substitutes for these document-specific calculations and were not run for this review-only change.

## Original experiment evidence

Files under `source/` were recovered from the investigation's local scratchpad:

`/tmp/claude-1000/-home-mixer-gufo/0f0c8823-bf21-4871-9575-7e0e56a78514/scratchpad`

They are retained to make the review's claims inspectable, **not newly authored or rerun benchmarks**:

- DS4 and Qwen KL logs preserve cumulative statistics and final summaries.
- `gguf_scan.json` preserves the original grouped inventory.
- `fp4_study_0731.py` and its JSON preserve the expert study's scale search and approximate Lloyd fitting. The referenced weight samples are not bundled.
- `shq_gemv_bench.hip` and `w4_unpack_wmma.hip` preserve the original microbenchmark source. Their layout/correctness/constant-scale limitations are reviewed explicitly; their historical rates were not reproduced here.
- `variants.sh` records the original tensor-selection recipe.

**Do not run archived source scripts as a reproduction command.** They contain original machine-specific paths, write outputs, and may convert/delete temporary models or invoke unsupported direct builds. Use the safe checks above. Reproducing throughput requires a separately corrected, correctness-checking harness, a Nix/release build appropriate to the repository workflow, and explicit model/runtime provenance.

`source-sha256.txt` hashes the archived files as included in this revision. Original log whitespace is retained (and is excluded from the authored-file whitespace check). Git pins these retained bytes; it does not retroactively pin the dirty executable used to generate the historical logs. The review distinguishes those original observations from newly checked arithmetic.
