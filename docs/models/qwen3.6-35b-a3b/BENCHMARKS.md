# Qwen3.6 35B-A3B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory (126976 MiB reported). Gufo
`560fcd0` (branch `feat/qwen35moe-35b-a3b`). Built on the host,
run inside the ROCm 7.2.3 container (the host's ROCm 7.1 cannot run Gufo).
Measured September 26, 2026, with the production inference services and the
OOM cron stopped so the benchmark owned the GPU.

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
(no 4 GiB chunking) loads the model cleanly with the production services
stopped; see [Experiments](EXPERIMENTS.md).

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

## Qwen3.8 27B regression (UD-Q8_K_XL)

Confirms the shared attention/GDN path is unchanged by the MoE work.

| Prefill prompt (tokens) | t/s |
| ---: | ---: |
| 2048 | 270.05 ± 0.94 |
| 8192 | 266.02 ± 0.03 |

## Reproduce

```sh
cmake --build --preset release --parallel 8
# stop the production services + OOM cron first (see the handoff)
for QUANT in Q8_K_XL Q6_K_XL; do
  /home/sami/gufo/run-in-container.sh bench \
    --model /home/sami/models-mtp/Qwen3.6-35B-A3B-MTP-UD-$QUANT.gguf \
    --n-prompt 512,1024,2048,4096,8192,16384 --n-gen 128 --repetitions 3
  for N in 1 2; do
    /home/sami/gufo/run-in-container.sh bench \
      --model /home/sami/models-mtp/Qwen3.6-35B-A3B-MTP-UD-$QUANT.gguf \
      --speculative mtp \
      --mtp-model /home/sami/models-mtp/Qwen3.6-35B-A3B-MTP-UD-$QUANT.gguf \
      --draft-tokens $N --min-draft-tokens $N \
      --n-prompt 512 --n-gen 128 --repetitions 3
  done
done
/home/sami/gufo/run-in-container.sh bench \
  --model /home/sami/models-gufo/qwen38-27b/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --n-prompt 2048,8192 --n-gen 0 --repetitions 2
```

Greedy, thinking off. [Quality and measurement scope](QUALITY.md).
