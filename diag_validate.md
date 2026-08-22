# Quantized decode+prefill correctness diagnostic — Qwen3.8-27B (Q8_K_L)

Env:
- BIN: /home/mixer/strix-halo.cpp/result/bin/strix-server (exists, exec)
- Model: /persist/models/.../Qwen3.8-27B-UD-Q8_K_L.gguf (symlink resolves)
- Binary: `result/bin/strix-server` is NOT a tracked git file (`git ls-files` empty; it is the nix `result` symlink/allowlist path). Present and executable.

## Command 1: validate (batched vs sequential prefill)
`timeout 150 $BIN bench --model $M -p 4 -n 4 --validate-prefill 4`
EXIT=0 (not timeout)

Log tail (diag_validate.log):
```
[Prefill Validation] tokens=4 sequential_top1=157 batched_top1=157 top1_match=yes finite=yes
  max_abs_diff=0.17779803 mean_abs_diff=0.02724309 rmse=0.03445510 cosine_similarity=0.99993390
| model                          |       size |     params | backend    | ngl |            test |                   t/s |
| ------------------------------ | ---------- | ---------- | ---------- | --- | --------------- | --------------------- |
| Qwen3.8-27B BF16               |  26.12 GiB |    27.32 B | ROCm (HIP) |  99 |             pp4 |          3.24 ± 0.00 |
| Qwen3.8-27B BF16               |  26.12 GiB |    27.32 B | ROCm (HIP) |  99 |             tg4 |          0.08 ± 0.00 |
```

NOTE: There is no `Qwen3.8-27B` plain row; table labels model as `Qwen3.8-27B BF16` (data-type label, see risks).

## Command 2: decode cost
`timeout 150 $BIN bench --model $M -p 1 -n 8`
EXIT=0 (not timeout)

Log (diag_decode.log):
```
| Qwen3.8-27B BF16               |  26.12 GiB |    27.32 B | ROCm (HIP) |  99 |             pp1 |          0.90 ± 0.00 |
| Qwen3.8-27B BF16               |  26.12 GiB |    27.32 B | ROCm (HIP) |  99 |             tg8 |          0.08 ± 0.00 |
```

## Verdict
- Cross-check: PASS. `top1_match=yes`, `finite=yes`, `cosine_similarity=0.99993`. `max_abs_diff=0.1778` is moderately elevated absolute logit divergence (mean 0.027, rmse 0.034) but is consistent with GPU batched-vs-sequential reduction ordering; no mismatch/FAIL emitted. Decode+prefill agree.
- Generator throughput: tg = 0.08 tok/s (both tg4 and tg8). Prefill: pp4=3.24 t/s, pp1=0.90 t/s. Generation is the ~0.08 tok/s bottleneck that makes the full -p 128 -n 16 run look hung; that workload was NOT run here (kept small by design).

## Residual risks
- Data-type label: bench reports `Qwen3.8-27B BF16` while size 26.12 GiB / 27.32B = 0.956 B/param, i.e. 8-bit (matches Q8_K_L). Label/type likely a reporting label, not necessarily the stored precision; verify if type label is load-logic or hardcoded.
- No profile/token breakdown printed beyond t/s table; per-kernel profile not captured by this bench output.
