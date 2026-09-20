# Qwen3.8-Flash-Next on Strix Halo

Linux x86-64, AMD `gfx1151`, 128 GB unified memory. Production builds with the
pinned Nix toolchain. PP/TG at d0/d32K/d128K: 2026-09-21; serving and vision:
2026-09-20; other CLI measurements: 2026-09-19. One repetition per point.
Target: `unsloth/Qwen3.8-Flash-Next-GGUF`
revision `38bb39ee97821de2c9009abb7e93950eec396e66`, `UD-Q4_K_XL` (four shards).
MTP: `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` from the same revision.

Chat follows the official template: thinking on, `xhigh` effort, prior reasoning
preserved. PNG/JPEG input works with AR/MTP and `mmproj-BF16.gguf`; see
[image usage and validation](README.md#images).
**Original unquantized-model and GGUF-conversion parity remain unqualified.**

## Single user

Tok/s, pp2048/tg128, greedy, seed 1, context capacity 133121. Depth precedes
the timed operation. The raw CLI uses a fixed repeated token pattern, warms
prefill and continues past EOS; d0 TG starts from 16 tokens. MTP PP includes
predictor catch-up through all known successor tokens.

| Depth | AR pp2048 | MTP pp2048 | AR tg128 | MTP tg128 | MTP acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1537.5 | 1503.6 | 26.43 | 43.95 | 69.2% |
| 4K | 1455.1 | TODO | 25.48 | 40.35 | 65.6% |
| 8K | 1432.0 | TODO | 25.25 | 35.68 | 58.3% |
| 12K | 1421.4 | TODO | 25.19 | 52.03 | 85.7% |
| 16K | 1412.3 | TODO | 24.91 | 56.31 | 88.5% |
| 32K | 1406.4 | 1380.3 | 24.28 | 65.65 | 100.0% |
| 64K | 1342.7 | TODO | 23.08 | 59.67 | 100.0% |
| 128K | 1310.5 | 1287.5 | 22.33 | 61.19 | 100.0% |

AR prefill falls 14.8% from d0 to d128K. MTP adds about 2% to prefill time
at the three refreshed depths, and its greedy output matches AR at each depth.
Single runs vary by about 1% within a session and the machine drifted 2%
across one evening; the d0 prefill change was qualified by interleaved runs of
both Nix binaries (1559.6 → 1568.1 tok/s). The 1700 tok/s target and flat
deep-context throughput remain unmet.

```sh
./result/bin/gufo bench --model "$MODEL" -p 2048 -n 128 \
  -d 0,4096,8192,12288,16384,32768,65536,131072 --temperature 0 --seed 1 -v
# Add --speculative mtp --mtp-model "$MTP" for MTP.
```

Sampled MTP, d0 raw prefix, seed 1, tg128, top-k/top-p/min-p disabled:

| Sampling | MTP tok/s | Acceptance |
| --- | ---: | ---: |
| T=0.7 | 39.83 | 65.2% |
| T=1.0 | 28.45 | 56.5% |

## Concurrent serving

**Sum of individual request decode rates**, in tok/s. Each rate uses
`completion_tokens / decode_seconds` from server-reported decode time;
prefill and waiting between decode steps are excluded. Context 4096, greedy,
thinking off, 128 output tokens, one cold cohort on a fresh server per point;
no cache hits. Set `--sessions C --max-pending-per-client 8` when benchmarking
up to eight requests from one host.
The [corpus](../qwen3.8-27b/artifacts/speculative-corpus.json)
uses `repetition_word` for repetition and distinct requests cycling through
`expository_pangram`, `cpp_ring_buffer`, `reasoning_train` for mixed work.

| Users | AR repetitive | MTP repetitive | AR mixed | MTP mixed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 27.3 | 84.6 | 27.0 | 49.3 |
| 2 | 46.4 | 127.7 | 43.2 | 66.4 |
| 4 | 76.7 | 166.1 | 66.8 | 95.2 |
| 6 | 95.4 | 181.2 | 78.2 | 110.4 |
| 8 | 108.5 | **188.9** | 86.7 | 118.3 |

MTP acceptance is 100% on repetition and 74.2–84.9% on the mixed cohorts.
Every completion matches AR; all cohorts report zero cache hits.
Greedy timing-based depth choices can vary between runs.
MTP rollback storage grows on demand to about 147 MiB per session at seven
drafts; KV caches and other session state are additional.

**C1 is a single user.** The HTTP and raw CLI prompts above differ.
Compare identical prompts and timing boundaries.
`gufo bench` is C1; use `tools/serving/gufo-serving-bench.py` for concurrency.

## Image encoder

Warm `mmproj-BF16.gguf` encoding; excludes preprocessing, first weight upload
and language-model prefill. Complete embeddings are byte-identical to the
previous encoder and reproduce exactly across runs.

| RGB image | Merged image tokens | Encode latency |
| --- | ---: | ---: |
| 256×256 | 64 | 21.3 ms |
| 1024×1024 | 1024 | 1249 ms |

The 4096-patch attention specialization saves 24 MiB of score/probability
scratch. Other image shapes retain their original tile layout.
