# Operating guide: optimizing the Qwen HRX backend

This file is meant to be handed to a fresh session that is asked to make
`src/models/qwen/hrx` faster. It records how to work on this target, what the
measurements mean, and which mistakes are already paid for. It is not a status
page — read these for state:

- `benchmarks/qwen3.8-27b/README.md`, section **"HRX native backend (Loom)
  experiments"** — every retained and rejected experiment, with numbers.
- `ideas.md` — the ranked open cards, each with the evidence it rests on.
- `task-on-going.md` — the running scratch log. Append to it as you go.

Read the two skill/agent files first: `AGENTS.md` and
`.agents/skills/optimize-kernel/SKILL.md`. This guide refines them for HRX
specifically; where they disagree about HRX, this file is the later evidence.

## What you are optimizing

`--qwen-backend hrx-native` is a second, independent backend. Its kernels are
Loom source in `tools/loom/*.loom`, compiled ahead of time by the toolchain
vendored through `.devops/nix/hrx-system.nix`; its executor is
`src/models/qwen/hrx/`. It shares the GGUF reader, tokenizer and sampler with
the HIP backend and nothing else.

**HIP is the baseline.** `--qwen-backend hip` on the same model is the number
to beat.

Two facts that should shape any plan:

- **Prefill is the blocked W8A8 projection**, about 76% of a 2048-token pass.
  The DeltaNet recurrence is ~7%, the attention kernel ~4.6%, and everything
  else is single digits. A change that does not touch the projection is
  arithmetically capped at a few percent.
- **Decode already beats HIP** (`tg16` ~7.90 against ~7.78). Do not spend time
  there unless asked.

The model is `models/Qwen3.8-27B-Q8_0.gguf`. Hidden 5120, FFN 17408, 64 layers
(48 SSM + 16 full attention), vocabulary 248320.

## Build and run

```sh
nix build .#hrx                       # release binary at ./result/bin/gufo
MODEL=models/Qwen3.8-27B-Q8_0.gguf
FUSIONS=blocked-prefill,swiglu-quant,norm-quant,readout-quant,paired-k

./result/bin/gufo bench --model "$MODEL" --qwen-backend hrx-native \
  --hrx-fusions "$FUSIONS" -p 2048 -n 0
./result/bin/gufo bench --model "$MODEL" --qwen-backend hip -p 2048 -n 0
```

`git add` before `nix build` — Nix only sees tracked files. Always `unset
LD_PRELOAD`; a stale one pointing at an old `libamdhip64.so` has bitten this
repo before.

New work goes behind a flag in `QwenHrxExecutionPolicy`
(`src/models/qwen/hrx/qwen_hrx_policy.{hpp,cpp}`), default off, so the old
route stays runnable as the comparison arm. Add a parse test in
`tests/models/qwen/qwen_hrx_model_contract_test.cpp` next to the existing ones.

## Measuring: the part that actually matters

Most wrong conclusions in this project came from measurement, not from kernels.

**Only interleaved, same-binary A/B is attributable.** Build one binary with
both routes behind a toggle, alternate the toggle, three rounds, report
medians and all raw samples:

```sh
for i in 1 2 3; do for f in "$BASE" "$CAND"; do
  v=$(./result/bin/gufo bench --model "$MODEL" --qwen-backend hrx-native \
      --hrx-fusions "$f" -p 2048 -n 0 2>/dev/null | grep -oP 'pp2048: \K[0-9.]+')
  echo "$i $f $v"
done; done
```

**Never compare two numbers taken at different times.** HIP maps its weights
and gets faster as the page cache warms: its `pp2048` moved between 484 and 559
t/s across one session on an idle machine. The cross-backend ratio is a rough
position, never a measurement of a change.

**The first timed point in a process is not comparable to anything.** `pp128`
measured first reads anywhere from 274 to 310 t/s on process state alone. Put a
throwaway length in front of anything short: `-p 512,128,256`. This single
mistake produced a phantom 10% regression in this session.

**Run one GPU job at a time, and nothing else heavy.** A compiler build running
concurrently inflated isolated kernel timings by 30%. The isolated harness
scores host-wall queue completion, so it is sensitive to CPU load.

**Isolated timing is a filter, not a verdict.** `iree-benchmark-loom` re-reads
one hot weight matrix and is LDS-bound; the deployed FFN streams tens of GiB
and is bound differently. Changes have gone both ways between the two. Use
isolated to reject cheaply and to read registers/occupancy; decide end to end.

A useful rule of thumb from repeated experience: **this host cannot resolve
better than about ±0.5% on `pp2048`.** A predicted win smaller than that is not
worth building unless it is free.

## Correctness

```sh
./result/bin/gufo bench --model "$MODEL" --qwen-backend hrx-native \
  --hrx-fusions "$FUSIONS" --validate-hrx 4 -p 1 -n 0
```

This compares HRX logits against a HIP reference for a batched prefill phase
and then per token. It is **route-aware**: the batched prefill phase uses a
W8A8 envelope when the policy enables the quantized prefill, and the tight f32
envelope otherwise; per-token phases always use the tight one. The phase line
prints which envelope it used.

It passes today. **A failure is a real failure** — this was not true before
2026-08-29, when the gate reported `envelope=fail` on every run and was
therefore checking nothing.

Aim for **bit-identical**: most useful changes here are traffic or scheduling
changes, and they should reproduce `rmse 0.04735499` / `cosine 0.99990022` /
top-1 157 exactly. If your numbers move, you have changed the arithmetic —
that may be fine, but it is now a decision that needs stating, not a detail.

Full gates before wrapping up:

```sh
nix build .#checks.x86_64-linux.tests
nix build .#checks.x86_64-linux.hrx-compile
```

`format` and `static-analysis` **already fail on a clean checkout** (in
`src/cli/bench/bench.cpp`, which HRX work does not touch). Verify that with a
stashed tree before blaming your change, and do not try to fix the whole repo
as a side quest. Format only the lines you touched:

```sh
ranges=$(git diff -U0 HEAD -- "$f" | grep -oP '^@@ -\S+ \+\K[0-9]+(,[0-9]+)?' \
  | awk -F, '{s=$1; n=(NF>1?$2:1); if(n>0) printf "--lines=%d:%d ", s, s+n-1}')
nix develop -c clang-format -i $ranges "$f"
```

## Profiling

**`rocprofv3` cannot profile the HRX executable.** It aborts inside
`hsa_executable_freeze` during IREE AMDGPU device init, before any model work.
Do not spend time on it. Hardware counters are unavailable too: gfx1151 is
11.5.1 and the counter path accepts gfx11 only for `minor == 0 && stepping <= 2`.

What works instead:

| Instrument | Gives you |
| :--- | :--- |
| `GUFO_HRX_TRACE_STAGES=1` | per-chunk attention / SSM / FFN split |
| `GUFO_HRX_TRACE_SSM=1`, `GUFO_HRX_TRACE_FFN=1` (layer 0), `GUFO_HRX_TRACE_ATTENTION=1` (layer 3) | per-substage split inside one layer |
| `loom-compile --compile-report=details --compile-report-output=r.json` | final VGPR/SGPR, scheduled pressure, occupancy tier, spills, `move_causes`, `static_instruction_mix` |
| `iree-benchmark-loom f.loom --device=amdgpu --benchmark=@<entry>_benchmark` | correctness-gated isolated timing |
| `llvm-objdump -d --mcpu=gfx1151` on `--emit-target-artifact` output | the actual ISA |

Every stage trace inserts a stream synchronization, so a traced run is slower
than the headline run, and **layer 0's first stage absorbs the previous stage's
drain** — its `norm` reads as tens of milliseconds and is not a real cost. Read
the *shape* from a traced run and the *throughput* from an untraced one.

To read the projection's hot loop, find the backward branch and slice between
its target and itself:

```sh
llvm-objdump -d --mcpu=gfx1151 k.hsaco > k.s
grep -n 's_branch 6[0-9]*' k.s | head -1      # target address is in the comment
```

The loop is currently 443 instructions: 16 `v_wmma`, 64 `v_cvt_f32_i32`,
64 `v_fma_f32`, 53 `v_mov_b32`, 31 `v_dual_mul_f32`, ~30 address VALU, and
151 SALU of which 78 are `s_delay_alu`.

## Kernel facts you should not rediscover

- **Occupancy tiers dominate.** 192 VGPRs is the 8-waves-per-SIMD tier and the
  projection sits at 184. Every variant that spent registers to save
  instructions lost, and several lost badly. Check
  `target_resources.vector.final.register_count` and
  `resident_subgroups_per_simd` in the compile report *before* benchmarking.
- **The 128x128 macro tile is a sharp local optimum.** Both dimensions are
  pinned. Traffic arguments lose to the occupancy cliff, and total DRAM traffic
  is minimized when the row and token spans are equal.
- **LDS is the second occupancy currency, and it is often the cheaper one.**
  Eight waves per SIMD needs four resident workgroups, so the budget is
  131072/4 = **32768 bytes per workgroup**; the hard per-workgroup limit is
  65536. The projection sits at 32256. Buying a kernel change with LDS instead
  of registers is what made card 2 work: the same change carried in registers
  cost 216 VGPRs, 7 waves and -5.6% at `pp2048`, and in LDS cost nothing and
  gained +4.4%. Read `resident_subgroups_per_simd` from the compile report --
  it accounts for both.
- **WMMA issues on the same SIMD32 vector ALUs as ordinary VALU**, so matrix and
  vector work add rather than overlap. Every epilogue instruction is paid for in
  matrix throughput. The measured int8 WMMA ceiling is 55.07 TOPS; f16 WMMA runs
  at half the int8 rate on this part.
- **Contiguity per thread per visit is what the weight stream costs.** Reading
  L contiguous bytes at arbitrary alignment costs `(128 + L - 1)/L` bytes of
  line traffic per useful byte. The projection's staging thread was at L=34
  (4.74x) and is now at L=136 (1.93x). This is the whole of card 2.
- **Fold the quantizer into whatever produces its input.** This pattern paid
  three times (SwiGLU, RMSNorm, DeltaNet readout) and is always bit-identical:
  on the blocked route the f32 tile has exactly one consumer, so writing it and
  reading it back is pure traffic. Check the producer's decomposition first — it
  works cleanly when a workitem or a wave already owns a multiple of 32
  contiguous elements.
- **Scalar ops exist for everything.** `scalar.absf`, `scalar.roundevenf`,
  `scalar.fptosi`, `scalar.siluf` and friends are all in the dialect. Grep
  `hrx-system/loom/py/loom/dialect/*/defs.py` for the real vocabulary rather
  than inferring it from what this repo happens to use — that mistake killed a
  good design once.
- The `index` address prover needs help. When a grid is rounded up past the
  logical count, give it `index.assume ... [range(...)]` or derive the capacity
  from the config symbol, as the existing kernels do.

## Changing the Loom compiler

It is vendored, and patching it is legitimate: add a patch file under
`.devops/nix/patches/` and a `patches = [ ... ];` line in
`.devops/nix/hrx-system.nix`. A full toolchain build takes roughly ten minutes.
Keep a local clone at `hrx-system/` checked out at the pinned revision for
reading.

Do this only with a hypothesis and an instrument. The productive loop is: add a
`fprintf` probe to the failing decision, build, read what the compiler actually
thinks, then write the real change. Guessing costs ten minutes a try and this
session burned eight builds guessing before probing.

Upstream moves daily. The pin is the newest revision that works here; `main`
currently fails because it queries `HSA_AMD_AGENT_INFO_PM4_EMULATION`, which
ROCr 7.2.3 rejects, leaving the AMDGPU accelerator unavailable. Check
`hrx-info` after any bump — it reports the accelerator in one line.

## Workflow

Use `jj`, not `git`. **One revision per experiment**, including rejected ones,
with the evidence in the commit or in the docs. Keep code and docs in separate
revisions:

```sh
jj new -m "perf(hrx): <change> (#200)"
jj squash --from <rev> --into <rev> <paths>   # to split a mixed revision
jj bookmark set main --revision <rev>
jj bookmark set fedeizzo/hrx-integration --revision <rev>
```

**Move both bookmarks, not just `main`.** `fedeizzo/hrx-integration` is the
branch this work is actually delivered on, and leaving it behind on an older
revision is easy to miss because everything else looks finished. It has tracked
`main` exactly so far, so the move is a fast-forward; check that before moving
it, and if `jj log -r 'main..fedeizzo/hrx-integration'` is not empty, stop and
ask rather than stranding whatever is only on that branch.

Do not push. Both bookmarks track `@origin` and both are deliberately left
behind it. Append findings to `task-on-going.md` as you get them — it is the
only artifact that survives a context reset mid-session.

**Rejection with evidence is a completed experiment**, and this codebase is
mostly rejections. When you reject, record the mechanism, not just the number,
so the next session does not rebuild it. Several of the most valuable results
here are negative: the VOPD `fmac` work was solved and *then* measured 4% slower,
which retired the largest card on the list.

## Autonomy

**Proceed without asking.** Do not stop to confirm a plan, to report a partial
result, or to ask which card to take next — pick the highest-value open card,
run the experiment, and keep going. Build, measure, decide, commit, move to the
next one.

Two consequences worth stating, because they are what "do not interrupt"
actually costs:

- **Decide from evidence, not permission.** Retain or reject on the measured
  A/B against the acceptance criterion in the card. If a change is neutral,
  reject it and say why. You do not need sign-off to reject.
- **Ask only when proceeding would be unsafe or wasted.** That means: a change
  that alters what the quantized model *is* (a precision reduction rather than
  a reassociation) needs an explicit decision and is not yours to make; and
  anything destructive or outside this repository needs confirmation. Note the
  question in `task-on-going.md`, do every part of the work that does not depend
  on the answer, and carry on.

Report at the end, not throughout. The end-of-run report should be the retained
and rejected list with numbers, not a narrative of what you tried.

## Before you start

1. Reproduce the baseline yourself, interleaved, and write it into
   `task-on-going.md`. Do not trust a number from a previous session; the host
   drifts.
2. Read the "closed" list at the bottom of `ideas.md`. It is there so you do not
   spend a day rediscovering that a 2x2 wave tile loses.
3. Pick a card with a stated size. If your change is predicted at under about
   0.5% of the pass, say so before building it, and expect the host not to
   resolve it.
