# Qwen3.6 35B-A3B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory (126976 MiB reported). Gufo
`560fcd0` (branch `feat/qwen35moe-35b-a3b`), run in the ROCm 7.2.3
`gufo-runtime` container. Measured September 26, 2026, with no other GPU
workload resident.

Prefill, generation and MTP decode are 3 repetitions; the 27B regression
check is 2.
The llama.cpp comparison uses the ROCm fork `llama.cpp-flashnext-hip`
b10672 (`llama-bench -ngl 99 -fa on -r 3`), measured the same day.

pp512 rows with a large ± include one slow warm-up repetition and are not
representative: an interleaved A/B of an earlier Q8_K_XL build measured
1294.77 ± 13.30 t/s at pp512, the MTP runs below measured 1589–1781 t/s at
pp512 with a small ±, and llama.cpp's Q6_K_XL pp512 (809.83 ± 223.14)
is affected the same way. Treat pp512 gains as indicative only.

## Qwen3.6 35B-A3B, UD-Q8_K_XL

| Prefill prompt (tokens) | pp (t/s) |
| ---: | ---: |
| 512 | 1475.15 ± 157.04 |
| 1024 | 2101.67 ± 5.84 |
| 2048 | 2311.31 ± 7.65 |
| 4096 | 2378.75 ± 2.63 |
| 8192 | 2299.53 ± 2.73 |
| 16384 | 2168.04 ± 3.30 |

| Workload | t/s |
| --- | ---: |
| tg128 | 51.59 ± 0.07 |

| Prompt (tokens) | gufo (t/s) | llama.cpp (t/s) | gain |
| ---: | ---: | ---: | ---: |
| 512 | 1475.15 (see note) | 1070.72 | +37.8% |
| 1024 | 2101.67 | 1061.29 | +98.0% |
| 2048 | 2311.31 | 1032.33 | +123.9% |
| 4096 | 2378.75 | 1003.12 | +137.1% |
| 8192 | 2299.53 | 960.41 | +139.4% |
| 16384 | 2168.04 | 881.82 | +145.9% |
| tg128 | 51.59 | 46.51 | +10.9% |

Cold mapped-weight load of the 36.65 GiB artifact to readiness is roughly
5–11 s depending on page-cache warmth. The single-shot `hipHostRegister`
(no 4 GiB chunking) loads the model cleanly when nothing else is resident;
see [Experiments](EXPERIMENTS.md).

## Qwen3.6 35B-A3B, UD-Q6_K_XL

| Prefill prompt (tokens) | pp (t/s) |
| ---: | ---: |
| 512 | 1628.88 ± 271.11 |
| 1024 | 2305.16 ± 0.83 |
| 2048 | 2571.00 ± 24.24 |
| 4096 | 2692.13 ± 8.97 |
| 8192 | 2586.20 ± 4.24 |
| 16384 | 2411.67 ± 1.11 |

| Workload | t/s |
| --- | ---: |
| tg128 | 54.02 ± 0.03 |

| Prompt (tokens) | gufo (t/s) | llama.cpp (t/s) | gain |
| ---: | ---: | ---: | ---: |
| 512 | 1628.88 (see note) | 809.83 (see note) | +101.1% |
| 1024 | 2305.16 | 942.11 | +144.7% |
| 2048 | 2571.00 | 930.46 | +176.3% |
| 4096 | 2692.13 | 920.41 | +192.5% |
| 8192 | 2586.20 | 867.42 | +198.1% |
| 16384 | 2411.67 | 800.98 | +201.1% |
| tg128 | 54.02 | 48.83 | +10.6% |

## MTP decode

| Quant, draft tokens | tg128-mtp (t/s) | tg128 without MTP (t/s) | gain |
| --- | ---: | ---: | ---: |
| UD-Q8_K_XL, n=1 | 60.23 ± 0.08 | 51.59 | +16.7% |
| UD-Q8_K_XL, n=2 | 65.01 ± 0.08 | 51.59 | +26.0% |
| UD-Q6_K_XL, n=1 | 63.53 ± 0.21 | 54.02 | +17.6% |
| UD-Q6_K_XL, n=2 | 68.39 ± 0.54 | 54.02 | +26.6% |

MTP uses the draft head embedded in the same GGUF (`--speculative mtp
--mtp-model <same file> --draft-tokens N --min-draft-tokens N`). Acceptance
depends on the generated text; these numbers use the benchmark's synthetic
prompt. The draft step costs about the same on both quants; an earlier
UD-Q8_K_XL run favoured n=1, so treat the n=1/n=2 gap on that quant as
run-dependent.

## llama.cpp Vulkan comparison

The ROCm fork above is not llama.cpp's fastest backend on this chip. Against
upstream llama.cpp `b81c99b47` on Vulkan (RADV, Mesa 26.1.7, `-fa 1`, no mmap;
prefill is the better of the default ubatch and `-ub 2048 -b 2048`), measured
September 27, 2026 with the same GGUFs, 3 repetitions:

| Test | UD-Q6_K_XL gufo | llama.cpp | gain | UD-Q8_K_XL gufo | llama.cpp | gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pp512 | 1789.6 | 1074.2 | +66.6% | 1483.6 | 1114.1 | +33.2% |
| pp2048 | 2579.7 | 1230.6 | +109.6% | 2325.7 | 1256.7 | +85.1% |
| pp4096 | 2702.1 | 1202.0 | +124.8% | 2383.6 | 1225.4 | +94.5% |
| pp16384 | 2409.2 | 1057.1 | +127.9% | 2175.1 | 1072.3 | +102.8% |
| tg128 | 54.5 | 56.4 | −3.3% | 50.8 | 46.2 | +9.9% |
| tg128 @ d16384 | 47.6 | 50.8 | −6.3% | 45.5 | 43.2 | +5.3% |
| tg128 @ d32768 | 42.3 | 46.6 | −9.4% | 40.5 | 39.7 | +2.0% |

Plain UD-Q6_K_XL decode trails Vulkan llama.cpp by 3-9%.

## Serving with DFlash2

`gufo serve` with the concurrent draft cap and BF16 context injection
(`46ac1d5`, `7659c0c`; measured before the rebase onto `8a46cd9`), 4 sessions,
context 262144, DFlash2 Q8_0 draft,
`--draft-tokens 7 --draft-policy adaptive`), UD-Q6_K_XL, over HTTP with a
synchronized client, thinking off. Measured September 27, 2026.

Single-request prefill (fresh random prompt per request, one output token),
before and after moving the draft's context injection to the BF16 WMMA GEMM:

| Prompt (tokens) | no draft | DFlash2 before | DFlash2 after |
| ---: | ---: | ---: | ---: |
| 520 | 1882 | 1451 | 1752 |
| 1630 | 2672 | 1871 | 2357 |
| 6420 | 2706 | 2104 | 2521 |
| 16000 | 2495 | 2076 | 2386 |

Concurrent decode, 512 sampled tokens (temperature 0.6, top-p 0.95, top-k 20)
from four mixed prompts (story, TCP explanation, Python module, Markdown
table), rotated so each level sees the same mix. Rates are the sum of
individual request decode rates per concurrent group, averaged over groups;
the concurrent draft cap shortens drafts to 3 / 2 / 1 tokens at 2 / 3 / 4
active requests:

| Concurrent requests | DFlash2 (t/s) | no draft (t/s) |
| ---: | ---: | ---: |
| 1 | 62.7 | 53.9 |
| 2 | 94.1 | TODO |
| 3 | 98.8 | TODO |
| 4 | 109.4 | 108.4 |

Without the cap (7-token drafts), 4 concurrent requests reached 88.7 t/s in
whole-request throughput, against 106.1 with it.

Prefill while other sessions decode is bounded by `--prefill-chunk`. A
6.5K-token prompt arriving while one session streams:

| `--prefill-chunk` | new prompt TTFT (s) | its prefill (t/s) | longest decode stall (s) |
| ---: | ---: | ---: | ---: |
| 512 (default) | 4.20 | 1862 | 0.37 |
| 1024 | 3.37 | 2190 | 0.52 |
| 2048 | 3.09 | 2330 | 0.89 |

## Qwen3.8 27B regression (UD-Q8_K_XL)

Confirms the shared attention/GDN path is unchanged by the MoE work.

| Prefill prompt (tokens) | t/s |
| ---: | ---: |
| 2048 | 270.05 ± 0.94 |
| 8192 | 266.02 ± 0.03 |

## Reproduce

With no other GPU workload resident, and `MODELS` pointing at the
directory holding the GGUFs:

```sh
cmake --build --preset release --parallel 8
for QUANT in Q8_K_XL Q6_K_XL; do
  MODEL="$MODELS/Qwen3.6-35B-A3B-MTP-UD-$QUANT.gguf"
  ./build/release/gufo bench --model "$MODEL" \
    --n-prompt 512,1024,2048,4096,8192,16384 --n-gen 128 --repetitions 3
  for N in 1 2; do
    ./build/release/gufo bench --model "$MODEL" \
      --speculative mtp --mtp-model "$MODEL" \
      --draft-tokens $N --min-draft-tokens $N \
      --n-prompt 512 --n-gen 128 --repetitions 3
  done
done
./build/release/gufo bench --model "$MODELS/Qwen3.8-27B-UD-Q8_K_XL.gguf" \
  --n-prompt 2048,8192 --n-gen 0 --repetitions 2
```

The verification-step and prefill-chunk timing tools are
`qwen35ba3b_verify_bench` and `qwen35ba3b_prefill_chunk_bench` (`gpu-test`
preset, `GUFO_QWEN35BA3B_MODEL`).

Greedy, thinking off. [Quality and measurement scope](QUALITY.md).
