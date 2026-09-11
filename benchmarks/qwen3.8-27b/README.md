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

The [quality report](eval/README.md) records the maintained checks, pinned
DFlash2 formulas, evidence and remaining gaps. Start with
`nix develop -c python3 tools/qwen27b/check.py fast`, then run only the affected
kernel/model suites. Release speed qualifies only after complete quality runs.

Production execution has one implementation per supported shape/weight type;
Qwen kernel, precision and verification environment switches are removed.
DFlash2 retains FP32 values in a bounded history ring. MTP replay uses the
committed target features. The optimized correctness build keeps assertions
and symbols; benchmark only the Nix release binaries.

Remaining: independent original-target/MTP qualification, final Q4/Q8 draft
selection, matched speed matrices, and speculative serving parity for C>1.
