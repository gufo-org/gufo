# 14 · Flash-Next MTP and AR adapter

**Milestone:** Adapters · **Depends on:** 02, 06, 08 · **Size:** L (split:
inventory, then AR, then MTP) · **Affects:** nothing served until card 19 ·
**Status:** proposed

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

### Adapter behavior after device loss

Context: this note comes from an out-of-tree model port that uses exact
recurrent checkpoints.

The adapter contract does not define teardown after a device loss. These
statements are in `src/cache/adapter.hpp` at `9024101b`:

- `Invalidate` returns false while any read or load is pending (lines 111-115).
- Destruction drains all outstanding slot transfers before the nonthrowing
  release path (lines 113-114).
- `BeforeRelease` is `noexcept` (line 30). Invalidation and destruction "cannot
  refuse release" (line 28). The guard must preserve or retire every borrower
  and finish all preservation transfers and readers before it returns
  (lines 28-30).

The RFC covers only the request side (RFC.md:329-331): "A device failure that
invalidates execution follows the normal request failure path; it cannot be
hidden as a harmless cache miss." No text defines `Invalidate`, release, drain or
teardown on a lost device.

Card 08 transfer pools and card 09 idle-spill workers keep copies in flight
beside decode. A fault can occur while transfers are pending. In that state,
`Invalidate` returns false. The only remaining path is destruction, and that
path waits on device work. Card 08 latches wait failures when HIP reports an
error. The observation below shows that a failure path with device calls can
also hang.

Observation on gfx1151 (Strix Halo) with ROCm 7.1.1:

- The hang occurred with the injected HIP errors of the device_loss functional
  suite. The suite injects HIP errors. It never resets the GPU.
- The failure path of the port made device calls. It called a session reset.
- A host-only invalidate, with no device calls, fixed the hang. The suite then
  passed 10 of 10 cases.
- The fix was not tested again on ROCm 7.2.3. Cards 08 and 09 measured on that
  version.
- On the same hardware, kernel compute-ring timeouts reset the compute queues
  during long GPU jobs ("device wedged"). This was observed with a long-running
  inference job. Device loss is therefore a real event, not only an injected
  one.

Request: add a device-lost state to the adapter contract (card 02 API). Card 14
applies it first. The rule applies to every adapter, not only to one model.
After the adapter latches the state:

1. `Invalidate`, `BeforeRelease` and destruction retire borrowers. They do not
   preserve them.
2. These calls do not wait on device work.
3. The cache treats all outstanding completions on that device as failed.
4. Transfer and spill workers stop. They do not block server shutdown or
   restart.
5. `tests/functional/device_loss.py` gains a case through its fault library. A
   spill or preservation copy is in flight when the injected loss arrives. The
   server must exit with status 75 without a hang.
