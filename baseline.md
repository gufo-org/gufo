# ROCm 7.2.3 Baseline — Qwen3.8-27B UD-Q4_K_XL

Reference measurement taken before any ROCm 10 packaging work, so the same
sweep can be replayed against a ROCm 10 build and compared directly.

## Provenance

| Field | Value |
| --- | --- |
| Date | 2026-09-04 |
| Repository revision | `9008a5e` (clean tree) |
| Toolchain | `nix eval .#default.toolchain --json` |
| ROCm | 7.2.3 (`rocmPackages.clr`) |
| Device HIP compiler | AMD Clang 22 |
| Host compiler | `gcc-wrapper-15.3.0` |
| GPU | gfx1151, AMD Radeon 8060S Graphics, wave size 32, 126976 MiB reported VRAM |
| Host kernel | Linux 7.1.8 |
| Build | `nix build` (default package, gfx1151 + XRT), exit 0 |
| Model | `models/Qwen3.8-27B-UD-Q4_K_XL.gguf`, 16.35 GiB, 27.32 B params |
| Model load | 1.43451 s |

## Command

```sh
nix build

./result/bin/gufo bench \
  --model models/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --n-prompt 2048 \
  --n-gen 128 \
  --n-depth 4096,8192,12288,16384 \
  --repetitions 1
```

## Results

Backend ROCm (HIP), `ngl 99`, single repetition, so the reported deviation is
0.00 by construction and these are single-sample points, not distributions.

| Test | t/s |
| --- | --- |
| pp2048 @ d4096 | 447.15 |
| tg128 @ d4096 | 11.46 |
| pp2048 @ d8192 | 426.25 |
| tg128 @ d8192 | 11.26 |
| pp2048 @ d12288 | 410.09 |
| tg128 @ d12288 | 11.06 |
| pp2048 @ d16384 | 392.43 |
| tg128 @ d16384 | 10.86 |

Depth scaling across 4096 → 16384: prefill falls 12.2% (447.15 → 392.43),
decode falls 5.2% (11.46 → 10.86).

## Notes

- The `model` column prints `Qwen3.8-27B BF16` while the shard is UD-Q4_K_XL.
  The size and parameter columns (16.35 GiB / 27.32 B) confirm the intended
  shard was loaded; the label comes from GGUF file-type metadata and is not
  evidence of a BF16 route.
- Single repetition was chosen to match the canonical depth sweep in
  `benchmarks/qwen3.8-27b/README.md`. Repeat with `--repetitions 3` if a
  ROCm 10 comparison lands inside a few percent of these numbers, since a
  single sample cannot separate that from thermal drift on a shared APU.
- No speculative decoding companion was configured for this run, so these are
  plain target-model numbers.
