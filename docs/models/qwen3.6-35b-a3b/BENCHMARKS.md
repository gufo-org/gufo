# Qwen3.6 35B-A3B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory (126976 MiB reported). Gufo
`bec9787` plus the uncommitted Qwen3.6-35B-A3B MoE work. Built on the host,
run inside the ROCm 7.2.3 container (the host's ROCm 7.1 cannot run Gufo).
Measured September 26, 2026, with the production inference services and the
OOM cron stopped so the benchmark owned the GPU.

Prefill and generation are 3 repetitions; the 27B regression check is 2.
The llama.cpp comparison uses the ROCm fork `llama.cpp-flashnext-hip`
b10672 (`llama-bench -ngl 99 -fa on -r 3`), measured the same day.

pp512 rows with a large ± include one slow warm-up repetition and are not
representative: an interleaved A/B of the same Q8_K_XL build measured
1294.77 ± 13.30 t/s at pp512, and llama.cpp's Q6_K_XL pp512 (809.83 ± 223.14)
is affected the same way. Treat pp512 gains as indicative only.

## Qwen3.6 35B-A3B, UD-Q8_K_XL

| Prefill prompt (tokens) | pp (t/s) |
| ---: | ---: |
| 512 | 1188.74 ± 465.09 |
| 1024 | 1793.14 ± 1.92 |
| 2048 | 1963.22 ± 10.27 |
| 4096 | 2029.74 ± 7.10 |
| 8192 | 1981.57 ± 2.17 |
| 16384 | 1875.30 ± 1.17 |

| Workload | t/s |
| --- | ---: |
| tg128 | 51.43 ± 0.02 |

| Prompt (tokens) | gufo (t/s) | llama.cpp (t/s) | gain |
| ---: | ---: | ---: | ---: |
| 512 | 1188.74 (see note) | 1070.72 | +11.0% |
| 1024 | 1793.14 | 1061.29 | +69.0% |
| 2048 | 1963.22 | 1032.33 | +90.2% |
| 4096 | 2029.74 | 1003.12 | +102.3% |
| 8192 | 1981.57 | 960.41 | +106.3% |
| 16384 | 1875.30 | 881.82 | +112.7% |
| tg128 | 51.43 | n/a | n/a |

Cold mapped-weight load of the 36.65 GiB artifact to readiness is roughly
5–11 s depending on page-cache warmth. The single-shot `hipHostRegister`
(no 4 GiB chunking) loads the model cleanly with the production services
stopped; see [Experiments](EXPERIMENTS.md).

## Qwen3.6 35B-A3B, UD-Q6_K_XL

| Prefill prompt (tokens) | pp (t/s) |
| ---: | ---: |
| 512 | 1319.28 ± 10.56 |
| 1024 | 1651.77 ± 32.85 |
| 2048 | 1891.08 ± 2.35 |
| 4096 | 2000.77 ± 6.76 |
| 8192 | 1953.46 ± 5.87 |
| 16384 | 1852.64 ± 0.53 |

| Workload | t/s |
| --- | ---: |
| tg128 | 53.95 ± 0.04 |

| Prompt (tokens) | gufo (t/s) | llama.cpp (t/s) | gain |
| ---: | ---: | ---: | ---: |
| 512 | 1319.28 | 809.83 (see note) | +62.9% |
| 1024 | 1651.77 | 942.11 | +75.3% |
| 2048 | 1891.08 | 930.46 | +103.2% |
| 4096 | 2000.77 | 920.41 | +117.4% |
| 8192 | 1953.46 | 867.42 | +125.2% |
| 16384 | 1852.64 | 800.98 | +131.3% |
| tg128 | 53.95 | 48.83 | +10.5% |

## Prefill after the October 2026 kernel changes

The routed down projection on the mixed pair map, the one-wave GDN post-norm
and QK norm, the fused residual/RMSNorm/quantize pass, the GDN convolution in
the qkv GEMM epilogue and the BF16 router and GDN controls (see
[Experiments](EXPERIMENTS.md)), measured October 2, 2026 as an interleaved A/B
against the source they were developed on. Both sides compile their HIP
sources with the pinned ROCm 7.2.3 compiler and run in the ROCm 7.2.3
container: `gufo bench --n-prompt 1024,2048,4096,8192,16384 --n-gen 0
--repetitions 3`, three rounds per artifact, mean of the rounds, production
inference services and the OOM cron stopped. The tables above predate the
routed F16 expert GEMM and are not comparable with these.

| Prompt (tokens) | UD-Q6_K_XL before (t/s) | after (t/s) | gain | UD-Q8_K_XL before (t/s) | after (t/s) | gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 2305.2 | 2632.4 | +14.2% | 2175.0 | 2408.3 | +10.7% |
| 2048 | 2629.0 | 2992.6 | +13.8% | 2415.4 | 2697.7 | +11.7% |
| 4096 | 2748.9 | 3153.9 | +14.7% | 2483.6 | 2804.1 | +12.9% |
| 8192 | 2667.9 | 3053.3 | +14.4% | 2420.7 | 2719.8 | +12.4% |
| 16384 | 2511.5 | 2849.8 | +13.5% | 2294.1 | 2559.5 | +11.6% |

Plain decoding is unchanged: tg128 on UD-Q6_K_XL is 55.5 t/s before and after.

Server mode, one session, greedy with thinking off: a 12,262-token prompt
prefills at 2533 → 2874 t/s on UD-Q6_K_XL and 2312 → 2580 t/s on UD-Q8_K_XL,
a 6,113-token prompt at 2645 → 3000 and 2402 → 2687 t/s. The completions are
not byte-identical to the previous build, because the router and the GDN
controls are rounded to BF16 in prefill; [Quality](QUALITY.md) has the checks
that qualified that change.

The compiler matters on its own: the same source built with the ROCm 7.1
compiler prefills about 2% slower (pp4096 2952 against 3004 t/s before the BF16
route) and decodes at 54.2 against 55.5 t/s.

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

## Serving

### MTP and DFlash2 on v0.17.1, October 2026

`gufo serve --sessions 4`, UD-Q6_K_XL, context 262144, `--prefill-chunk 2048
--prefill-decode-rounds 8`, rebased on upstream `ef15911`, measured October 19,
2026 over HTTP, greedy, thinking off, one run per row. Short prompts generate
300 tokens; the cold rows feed that many tokens of a source-code corpus with
a prose question and generate 400. DFlash2 uses the Q8_0 draft with
`--draft-tokens 7 --draft-policy adaptive`; MTP uses the GGUF's own head with
`--draft-tokens 2`. Every MTP output is byte-identical to the DFlash2 output
for the same prompt.

| Workload | DFlash2 prefill (t/s) | MTP prefill (t/s) | DFlash2 decode (t/s) | MTP decode (t/s) |
| --- | ---: | ---: | ---: | ---: |
| short prompts (code, prose, explain, JSON) | — | — | 117 / 60 / 85 / 136 | 82 / 62 / 76 / 86 |
| cold 4K | 2819 | 3143 | 75.3 | 69.5 |
| cold 16K | 2770 | 2841 | 67.1 | 67.3 |
| cold 32K | 2523 | 2671 | 63.7 | 62.9 |
| cold 64K | 2221 | 2203 | 63.8 | 56.6 |
| cold 128K | 1670 | 1671 | 54.4 | 45.1 |
| 2 / 4 concurrent short requests, total | — | — | 95.6 / 122.3 | 87.0 / 110.5 |

MTP skips the draft's context projection during prefill, which is why it
prefills short and medium prompts faster. Concurrent MTP requests verify in one
batched step: at 2 / 3 / 4 requests they total 87.0 / 99.0 / 110.5 t/s against
83.0 / 101.5 / 114.0 without a draft, with completions byte-identical to the
same prompts sent alone. At 2 requests one draft token beats two: 87.0 against
85.6 t/s on UD-Q6_K_XL, and 81.4 against 78.8 on UD-Q8_K_XL (74.7 without a
draft). Proposing the sessions' drafts in one batch, with one host
synchronization per round instead of one per draft token, measured within 0.2%
of the per-session proposals on this machine, with identical outputs. Before that, concurrent MTP sessions took the
single-token path, which rejects speculative sessions and does not advance the
draft: requests with stop strings, a JSON schema or the batch endpoint failed,
and some returned corrupted text.

The same day, on the same workload, the build before the rebase (identical
35B code, older base) swept the MTP draft cap. Decode t/s:

| Workload | no draft | MTP n=1 | MTP n=2 | MTP n=3 |
| --- | ---: | ---: | ---: | ---: |
| cold 4K | 54.5 | 65.9 | 69.4 | 66.6 |
| cold 16K | 51.9 | 63.2 | 67.2 | 64.0 |
| cold 32K | 48.7 | 58.0 | 62.7 | 61.7 |
| cold 64K | 43.5 | 52.4 | 52.6 | 55.6 |
| cold 128K | 36.3 | 43.3 | 45.0 | 42.5 |

n=2 is the best cap on this workload; n=3 only wins at 64K and on short code
and JSON answers.

### DFlash2, September 2026

`gufo serve` on the September branch, with the concurrent draft cap and BF16
context injection as they first landed (draft caps 3 / 2 / 1), 4 sessions,
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
| 2 | 94.1 | not measured |
| 3 | 98.8 | not measured |
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
| 2048 | 268.88 ± 0.57 |
| 8192 | 265.06 ± 0.53 |

## Reproduce

```sh
cmake --build --preset release --parallel 16
# Nothing else may use the GPU while these run.
for QUANT in Q8_K_XL Q6_K_XL; do
  ./build/release/gufo bench \
    --model "$MODELS/Qwen3.6-35B-A3B-MTP-UD-$QUANT.gguf" \
    --n-prompt 512,1024,2048,4096,8192,16384 --n-gen 128 --repetitions 3
done
./build/release/gufo bench \
  --model "$MODELS/Qwen3.8-27B-UD-Q8_K_XL.gguf" \
  --n-prompt 2048,8192 --n-gen 0 --repetitions 2
```

Greedy, thinking off. [Quality and measurement scope](QUALITY.md).
