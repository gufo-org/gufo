# 14 · Flash-Next MTP and AR adapter

**Milestone:** Adapters · **Depends on:** 02, 06, 08 · **Size:** L (split:
inventory, then AR, then MTP) · **Affects:** nothing served until card 19 ·
**Status:** draft implementation; qualification in progress

## Goal

Implement the adapter for Flash-Next in MTP and AR modes, and prove with
model-local tests that capture and restore reproduce the exact state. First
adapter per D2. Serving keeps using the legacy cache until card 19.

## Step 1: capacity and inventory (reviewable on their own)

- **Capacity.** About 114 GB of weights on 125 GiB of visible memory leaves a
  narrow margin. Confirm the RFC's MTP and AR capacity rows with committed
  backing (D4) included. Eight long live histories are excluded.
- **Inventory**, starting from the RFC table and the model's
  `SessionSnapshot`:

  | Component | Proposed kind |
  | --- | --- |
  | Target attention KV | append rows |
  | MTP draft KV | append rows, only where certified immutable; own position |
  | Pooled indexer rows | append rows, in completed blocks |
  | Raw indexer ring | private (chronological contents and cursor) |
  | GDN state and convolution history | private |
  | Kept hidden rows, residuals, execution metadata | private, per numerical contract |
  | Speculative bookkeeping | model-owned part private; request-owned part reset by serving |

- **Mutation paths to audit:** prefill, decode, MTP draft and verify, rollback
  after rejected drafts, reset, session destruction.

## Step 2: adapter

- Descriptors, `CapturePrivate`, row copies through card 08's primitives,
  `Validate`, and guard calls on every audited mutation path.
- **No cost while inactive:** until card 19 attaches the new cache, the guard
  calls do nothing on the production path.
- The runner state holds the model-level session the adapter owns (card 02
  review point).
- The model's existing borrowed-snapshot mechanism (`SnapshotMode::kBorrowed`
  and its shared backing blocks) stays until card 19 deletes it with the
  legacy cache. No two borrowing systems remain after the switch.

## Test first

Model-local tests:

- each component round-trips, including the ring wrapping around and pooled
  blocks that are not yet complete;
- next-step logits and MTP drafts after a restore into another slot match an
  uninterrupted run;
- borrowed rows are preserved before an overwrite on each mutation path;
- an edit before a checkpoint restores an earlier boundary and never
  truncates recurrent state.

## Step baseline

For MTP and AR, at 32k and 100k tokens: private bytes per checkpoint, row
bytes per token per component, capture time, and restore time into another
slot. RFC E1 for comparison: about 113.8 MiB fixed plus 27.46 KB per token
(MTP).

## Done when

- [ ] Tests above pass on gfx1151 for both modes.
- [ ] The model's standard speed benchmark against matched `main` shows no
  regression from the inactive guard calls (per phase, 5% / 3 ms).

## After this card

Run the card 19 branch locally for Flash-Next to get early HTTP evidence
(D1). That branch is not merged until card 19's checks pass.

## Review focus

- The raw indexer ring: is "private" right, or can completed windows be
  shared?
- Is the inventory complete? Missing state is silent corruption.

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) (ICLR 2025): GDN state semantics: why the state cannot be truncated.
- [Hybrid Models Meet SGLang: More than Full Attention](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/): recurrent-state checkpoints and speculative decoding for Qwen3-Next.
- [SGLang Flash-Next MTP reference](https://github.com/sgl-project/sglang/blob/993d1fccbaafe3e79d91567d2fc1d665cc94fa50/python/sglang/srt/models/qwen4_exp_mtp.py): MTP state the reference implementations keep.
- [vLLM Flash-Next MTP reference](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/models/qwen4_exp/amd/mtp.py): the same, in vLLM.
- [llama.cpp server slots and context checkpoints](https://github.com/ggml-org/llama.cpp/blob/41abbfd59/tools/server/server-context.cpp): context checkpoints for recurrent models.
- [Flash-Next prompt checkpoints](../../models/qwen3.8-flash-next/PROMPT-CHECKPOINTS.md) (in this repository): today's borrowed snapshots, their timings and the allocator decision.

## RFC

[Component descriptors and model modules](../RFC.md#component-descriptors-and-model-modules) ·
[Correctness contract](../RFC.md#correctness-contract) ·
[Per-model capacity matrix](../RFC.md#success-criteria-and-evaluation) (under Success criteria)

## Review notes


## Results (2026-10-10, draft)

The model-local [inventory and mutation audit](../../models/qwen3.8-flash-next/CONTINUATION-ADAPTER.md)
record 124 AR / 130 MTP components, private recurrent/ring state and independent
target/draft/pool frontiers. Serving still uses the legacy cache.

Environment: homelab, Linux 7.2.9, gfx1151, 124 GiB HIP-visible memory,
performance platform profile, pinned Nix GCC 15.3.0 / ROCm 7.2.3. Production
CMake release builds compare with clean main `92aaed5d30cd82e5730e43be3501775b77c26e5e`.
Weights are the retained four-shard UD-Q4_K_XL target and shared Q8_0 predictor
from [model identities](../../models/qwen3.8-flash-next/artifacts/model-identities.json).

CPU geometry passes. The earlier candidate passed AR/MTP exact component and
next-step checks, seeded rejection and carried residual/controller replay, empty
state, reset/release, peer-stream overlap, shorter/edit restore and image input
identity/layout. Legacy snapshot checks and independent GDN numerical checks
pass. Hosted CPU checks passed 54/54; the geometry test has since been added to
the hosted target and needs the final 55-test check. The final prefill-tail
planner and injected reset-submission failure checks are awaiting their rerun.

Production adapter baselines use serial transfers into packed, precommitted
64 MiB slabs, with 4 GiB total backing committed before prefill. Times exclude
model loading, backing allocation and byte-verification copies. Restore includes
`BeginRestore`, all component loads and `Validate`. Both destination bytes and
next-step logits/drafts are checked exactly.

| Mode | Tokens | Private bytes | Append bytes | Private capture ms | Full materialization ms | Cross-slot restore ms | Assigned slab bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AR | 32,768 | 131,614,880 | 830,603,264 | 8.35 | 19.89 | 12.68 | 1,006,632,960 |
| AR | 100,000 | 131,614,880 | 2,534,800,000 | 8.04 | 40.21 | 33.26 | 2,684,354,560 |
| MTP | 32,768 | 133,032,096 | 899,806,976 | 8.67 | 21.64 | 13.61 | 1,073,741,824 |

MTP 100k qualification is still running. The 32k/100k cross-slot probes use
slot capacities 32,896/100,128 so the restored next step fits. AR C1 at capacity
100k and AR C8 at capacity 8,192 also fit with every live slot prefilled, four
decode cycles, 4 GiB physically committed backing and conservative remaining
rollback/vision/scratch claims. These measurements qualify the adapter and
physical memory envelope, not card 19's checkpoint-store allocation policy.
Eight long histories remain excluded.

The initial inactive-hook speed gate passes at d0/32k for pp2048/tg128 AR and
MTP mixed/repetitive: counts, completion hashes, proposed drafts and acceptance
are exact. Per-phase time changes are within 2.79% prefill and 0.97% decode;
there is no averaging across requests. All setup observations were retained: the
first candidate had stale build dependencies, and one fresh MTP 32k observation
overlapped a test rebuild. Those timings are unqualified; clean matched repeats
pass. The final reset-error hardening still needs its production rerun.

Candidate HTTP `long-context` and `cache` correctness passes in AR (zero drafts)
and MTP (298 actual drafts, 210 accepted). Individual request/phase timing flags
in snapshot copies, restore, cancellation and queueing remain **inconclusive and
unqualified**; unchanged-main controls and the final candidate rerun are pending.
The new cache's HTTP path remains deferred until card 19 exists and attaches it.

Reproduction (select a mode and depth explicitly):

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen38_flash_next_continuation_adapter_test
build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_continuation_adapter_test \
  --model /persist/models/qwen38-flash-next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mtp-model /persist/models/qwen38-flash-next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf

nix develop -c cmake --preset release -DGUFO_BUILD_TOOLS=ON
nix develop -c cmake --build --preset release \
  --target qwen38_flash_next_continuation_probe
build/release/src/models/qwen38_flash_next/qwen38_flash_next_continuation_probe \
  --model /persist/models/qwen38-flash-next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mtp-model /persist/models/qwen38-flash-next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --mode mtp --context 100000 --sessions 2 --round-trip
```

Raw logs, commands, JSON request measurements and failed/inconclusive runs are
retained outside the repository under `/tmp/gufo-card14-*`; they will be copied
to persistent qualification storage before final review. Benchmark `source`
metadata identifies the harness checkout, so binary hashes and the clean main
workspace/source patch are recorded separately. No standalone measurement JSON
is added to this repository.
