# Qwen3.8 27B on Strix Halo

Production targets are **UD-Q4_K_XL and UD-Q8_K_XL**. Optional speculative
routes are MTP and DFlash2. BF16 is a quality reference, not a production
configuration. Linux x86-64, gfx1151, 128 GB unified memory; Nix release builds.

Qualification is in progress. `TODO` means no current qualified measurement.
Prefill uses **pp4096**, generation uses **tg128**, and `C` means concurrent
requests. Generation rates below are per user; aggregate rate is `C` times
that number. Draft precision is selected separately from target precision.

## Single user, autoregressive

| Context depth | Q4 pp / tg (tok/s) | Q8 pp / tg (tok/s) |
| ---: | ---: | ---: |
| 0 | TODO | TODO |
| 4,096 | TODO | TODO |
| 8,192 | TODO | TODO |
| 12,288 | TODO | TODO |
| 16,384 | TODO | TODO |

## Single user, speculative

Prefill / generation in tok/s. DFlash2 companion selection is pending the
matched Q4/Q8 comparison; MTP uses the separate Q4_0 artifact.

| Context depth | Q4 + DFlash2 | Q8 + DFlash2 | Q4 + MTP | Q8 + MTP |
| ---: | ---: | ---: | ---: | ---: |
| 0 | TODO | TODO | TODO | TODO |
| 4,096 | TODO | TODO | TODO | TODO |
| 8,192 | TODO | TODO | TODO | TODO |
| 12,288 | TODO | TODO | TODO | TODO |
| 16,384 | TODO | TODO | TODO | TODO |

## Multiple users, autoregressive

Cells are aggregate prefill / per-user generation, in tok/s.

| Context depth | Q4 C2 | Q8 C2 | Q4 C4 | Q8 C4 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | TODO | TODO | TODO | TODO |
| 4,096 | TODO | TODO | TODO | TODO |
| 8,192 | TODO | TODO | TODO | TODO |
| 12,288 | TODO | TODO | TODO | TODO |
| 16,384 | TODO | TODO | TODO | TODO |

## Multiple users, speculative

Cells are aggregate prefill / per-user generation, in tok/s. Batched
speculative serving is being qualified before these numbers are published.

| Context depth | Q4 C2 | Q8 C2 | Q4 C4 | Q8 C4 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | TODO | TODO | TODO | TODO |
| 4,096 | TODO | TODO | TODO | TODO |
| 8,192 | TODO | TODO | TODO | TODO |
| 12,288 | TODO | TODO | TODO | TODO |
| 16,384 | TODO | TODO | TODO | TODO |

## Draft choice

Compare companions against the **same target and chat-framed prompts**.
Q4 and Q8 targets can generate different continuations, so their acceptance
rates do not directly rank the companions. Each pairing must reproduce its
own target's greedy output and token count before its speed qualifies.

| Target | DFlash2 Q4_K_M | DFlash2 Q8_0 | Production choice |
| --- | --- | --- | --- |
| Q4_K_XL | TODO | TODO | TODO |
| Q8_K_XL | TODO | TODO | TODO |

## Reproduce

```sh
nix build
MODEL=/path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf
DRAFT=/path/to/Qwen3.8-27B-DFlash2-Q8_0.gguf
MTP=/path/to/mtp-Qwen3.8-27B-Q4_0.gguf

./result/bin/gufo bench --model "$MODEL" \
  -p 4096 -n 128 -d 0,4096,8192,12288,16384 -c 1 -r 2 -v
./result/bin/gufo bench --model "$MODEL" \
  --speculative dflash2 --dflash-model "$DRAFT" \
  -p 4096 -n 128 -d 0,4096,8192,12288,16384 -c 1 -r 2 -v
./result/bin/gufo bench --model "$MODEL" \
  --speculative mtp --mtp-model "$MTP" \
  -p 4096 -n 128 -d 0,4096,8192,12288,16384 -c 1 -r 2 -v
```

Use short probes while developing. Run hardware jobs sequentially, profile in
a separate pass, and alternate baseline/candidate release measurements.
Synthetic depth sweeps measure the engine; chat corpus comparisons measure
acceptance on useful workloads. Report both without combining their rates.

## Quality and feature parity

- Check operators against pinned official formulas and independent CPU
  equations. Comparing two Gufo paths alone can miss a shared error.
- Compare prefill and decode logits, finite values, greedy choices, and
  repeated output on each quantization. BF16 comparisons isolate quantization
  effects and stay in optional reference tests.
- Require speculative output and token counts to match ordinary decoding.
  Include short budgets, rejection/rollback, context boundaries, snapshots,
  prefix reuse, and concurrent requests. Failed or missing cases fail the run.
- Exercise `prompt`, `chat`, `bench` and HTTP serving consistently. Check
  sampling, bounded prefill, cancellation, context exhaustion and disk-cache
  identity against the same service contract used by DS4.

Remaining qualification: official MTP/DFlash2 audit, Q4/Q8 draft selection,
entrypoint parity, and matched C1/C>1 performance. The BF16 reference cannot
by itself establish equivalence to the original unquantized checkpoint.
