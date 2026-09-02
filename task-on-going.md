# task: dflash2 companion x target-quant matrix (fedeizzo/dflash2-comparison-qwen)

Question: which DFlash-2 companion is best per target quant, on speed AND acceptance.

## Grid
Targets:
- Q8 = models/Qwen3.8-27B-UD-Q8_K_L.gguf  (Q8_K_XL is a symlink to this exact file)
- Q4 = models/Qwen3.8-27B-UD-Q4_K_XL.gguf
Drafts:
- dQ4  = models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf  (1.14 GB)
- dQ8  = models/Qwen3.8-27B-DFlash2-Q8_0.gguf    (2.06 GB)
- dBF16= models/Qwen3.8-27B-DFlash2-BF16.gguf    (3.86 GB, created 2026-09-01)
6 combinations.

## Known state (benchmarks/qwen3.8-27b/README.md)
- Q8+dQ8 19.4-27.3 t/s (unstable), Q4+dQ4 18.2-18.4 t/s (stable), Q4+dQ8 14.4 t/s.
- AR tg128: Q8 6.89, Q4 11.59.
- draft-policy auto -> fixed width 7 for dflash2.
- Corpus runner: tools/quant/speculative-corpus.py, reruns AR per (case,invocation).

## Env / build
- result -> gufo-84a95ac (= main HEAD), clean.
- host load ~3.1 at start; README warns noisy host.

## Log

### Host confound (2026-09-01)
- pid 1366021 `gufo serve -v --port 9999 llm --model .../UD-Q4_K_XL.gguf` running since
  11:32, 8 GB RSS, 43% VRAM allocated, GPU busy 0%. Owned by the user, NOT killed.
  It is idle but holds unified memory; treat all absolute numbers as taken under it.
- Host load ~3.1 at start.

### Harness changes
- tools/quant/speculative-corpus.py: added --ar-cache (AR reference is a function of
  target+length+prompt only, so 6 companions share 1 reference per target),
  --json report, --label.
- tools/quant/dflash-matrix.sh: drives the 2x3 grid, rotates companion order per rep.

### Finding 1 (fixed): BF16 companion load was minutes of scalar work
`PackMatrixBf16` widened every source row to float and rounded back to BF16.
For a BF16 source that round trip is the identity (FloatToBfloat16Bits on a
value with zero low mantissa bits returns the same bits), but it ran scalar and
single-threaded over ~1.9G elements. Observed: `gufo prompt` on Q4 + DFlash2-BF16
spent >7 min at 99.8% CPU on one thread before emitting anything.
Fix: direct hipMemcpy when `source.type == kBF16`. Bit-identical.
Also added: `GUFO_DFLASH_DEBUG` now reports private packed draft bytes + pack seconds.

### Measurement plan
- tools/quant/dflash-matrix.sh drives 2 targets x 3 companions.
- Corpus: speculative-corpus.json (10 cases) at 128 tokens, width 7, policy auto.
- Then speculative-adaptive-corpus.json (3 cases).
- AR reference measured once per target (shared via --ar-cache).
- tools/quant/dflash-matrix-report.py pivots the JSON into the answer tables.

### Baseline load time, old binary (gufo-84a95ac)
`gufo prompt --model UD-Q4_K_XL --speculative dflash2 --dflash-model DFlash2-BF16 --max-tokens 8`
-> `[Model Load]: 446.701 s`, output sane, acceptance 11.4% over 5 steps.
This is the number the BF16 fast path has to move.

### Finding 1 result: BF16 fast path lands
Warm page cache, `gufo prompt --max-tokens 8`, target UD-Q4_K_XL, GUFO_DFLASH_DEBUG=1,
new binary (gufo-84a95ac-dirty):

| Companion | Private draft bytes | Pack seconds | Model load |
| --- | ---: | ---: | ---: |
| DFlash2-BF16   | 3,849,344,000 | 3.13 | 1.14 s |
| DFlash2-Q8_0   | 2,336,691,200 | 9.84 | 1.16 s |
| DFlash2-Q4_K_M | 1,575,900,160 | 5.06 | 1.16 s |

All three emit the same 8 tokens and the same acceptance (11.4%) on "Hello".
Cold first-touch of the BF16 companion on the OLD binary was `[Model Load]: 446.701 s`.
Note the inversion the fix creates: BF16 is now the FASTEST companion to pack (pure
memcpy) while Q8_0 is the slowest, because Q8_0's selector codebooks, fc_projection
and attn_k/attn_v still dequantize through the scalar per-element route -- those
matrices are forced to BF16 storage regardless of the artifact's format.

Private draft bytes is the number that should predict per-step draft bandwidth:
BF16 2.44x Q4_K_M, Q8_0 1.48x Q4_K_M.

### Finding 2 (written, not yet measured): draft non-causal attention QK stage is uncoalesced
`dflash_noncausal_attention_kernel` gives each key position ONE thread which then
walks head_dim=256 contiguous floats, so consecutive threads are kv_dim=1024 floats
apart and each 4-byte read pulls a full line. README already flags this as 16.26 ms
per block at 4K context against 0.34 ms at 128.
Added `dflash_noncausal_attention_wave_kernel`: one wave32 per key, float4
lane-strided loads, `__shfl_xor` reduction -- the same shape
`attention_batched.hip:130` already uses for the target. PV stage untouched (it is
already coalesced). Selected by `GUFO_DFLASH_ATTENTION=wave`, default stays scalar.
Needs: build, equivalence check, A/B at --n-depth 4096/8192.

### Finding 3 (host, not code): a cold target shard costs more than the whole sweep
`gufo prompt` on UD-Q8_K_L with a cold page cache reported `[Model Load]: 791.52 s`;
the identical command immediately after reported `6.565 s`. During the cold load the
process sits at 100% *user* CPU with `minflt`/`majflt` frozen after the first 39 s and
`read_bytes` flat, so it is not disk: the shard is 28 GB against ~100 GB of page cache
already held by the idle `gufo serve`, deluge and restic, and the registration walk
thrashes. UD-Q4_K_XL (17 GB) never reproduces it -- it loads in 1.67 s.

Consequence for the sweep: every 900 s timeout recorded on 2026-09-01 and in this
session's first smoke run was a cold load, NOT a property of the target/companion
pairing. `tools/quant/dflash-matrix.sh` now `cat`s the target and companion shards
before each arm and the default timeout is 1800 s.

### Harness changes (2)
- dflash-matrix.sh: shard prewarm, TIMEOUT 1800, suite name in the report tag so two
  corpora can share one output directory and one AR cache.
- dflash-matrix-report.py: `--pattern` to aggregate one corpus at a time.
