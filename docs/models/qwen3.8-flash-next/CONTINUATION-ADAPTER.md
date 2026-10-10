# Flash-Next continuation adapter (card 14)

`ContinuationAdapter` owns the public model `Session` used by the runner.
Serving still uses the legacy cache. Its sessions leave both hook pointers null;
card 19 will attach the adapter and remove the legacy borrowed snapshots.

## State representation

The caller supplies the full artifact, tokenizer and template identity. The
adapter adds its component ABI, legacy snapshot ABI, context, prefill capacity,
maximum speculative width, concurrency and AR/MTP policy. IDs and layouts are
stable for an adapter lifetime. Target and draft frontiers are independent.

| State | Representation | Bytes for the retained model |
| --- | --- | ---: |
| Tokens | append rows, int32 | 4/target token |
| Target attention K and V | separate f16 append components per attention layer | 1,024/row/component; 24,576/target token total |
| Target pooled indexer | f16 append rows per completed compression block | 256/block/layer; 768/target token at ratio 4 |
| Raw indexer ring | private chronological pending rows, zero padded to 2,048 rows | 1,048,576/attention layer |
| Convolution history | private f32, per linear layer | 122,880/layer |
| GDN recurrent matrix | private f32, per linear layer | 3,145,728/layer |
| PLE convolution history | private f32 | 368,640 |
| Next-step logits | private f32 | 993,280 |
| Execution metadata | private, fixed ABI | 160 |
| MTP K and V | separate f16 append components at draft frontier | 2,048/draft token |
| MTP pooled indexer | f16 append rows at draft completed-block frontier | 256/draft block |
| MTP raw indexer ring | private, same canonical format as target | 1,048,576 |
| MTP residual | private f32, zero when invalid | 40,960 |
| Kept target hidden rows | private f32, up to eight rows | 327,680 |

There are 124 components in AR and 130 in MTP. Append chunks contain 256
component rows, including pooled rows (not 256 target tokens). GDN and
convolution histories are exact-boundary private state; they cannot be truncated
when restoring a shorter target frontier. Raw ring contents stay private because
future writes wrap and partial compression blocks remain mutable. The cursor is
reconstructed from the saved frontier and completed-block count. Completed
pooled rows are shareable.

At context >= 2,048, private state is 131,614,880 bytes (125.5177 MiB) in AR
and 133,032,096 bytes (126.8693 MiB) in MTP. The slope is 25,348 bytes per
target token, plus 2,048 bytes per draft token and 256 bytes per completed draft
block. Fully caught-up MTP has a 27,460-byte/token slope. This includes token
IDs. The fixed private representation reserves the largest pending raw window,
which explains its additional space compared with the RFC's roughly 113.8 MiB
legacy deep-context payload. Unused private bytes are serialized as zeros.

Metadata includes PLE n-gram history, the per-session acceptance length
controller, hidden frontier, residual validity and image input identity. The
shared batch timing controller is execution workspace: restoring one session
must not rewind its peers. Sampler RNG/history, request counters, cancellation
and stop rules belong to the request and are not checkpoint state.

Vision layout is reconstructed from a matching immutable request attachment,
including a checkpoint inside image placeholders. A missing or different image
identity refuses validation. Restoring a text checkpoint clears an old image
attachment. Scratch, graph handles, rollback operands and temporary proposals
are not continuation state; captures require a committed frontier.

## Mutation and transfer contract

The hooks cover public prefill, evaluate, single and batched decode, vision
configuration, draft policy reset, legacy restore and reset. Native hooks cover
scalar and batched target/draft forwards before graph replay, speculative
rollback and MTP rewind. All append ranges are preserved before writes, including
completed pools; reset/restore preserve the entire replaceable range while it is
readable. Destruction preserves or retires every borrower before releasing
storage. Legacy borrowed-snapshot preservation remains installed independently.

Transfers use card 08's nonblocking streams and completion leases. Pending
captures prevent mutation; pending loads prevent execution and invalidation.
Private tensors support arbitrary byte pieces; row and private loads can arrive
out of order on independent streams. Validation requires complete byte coverage
and exactly one consistent frontier per component. Any load failure latches even
if its completion is discarded. A failed reset keeps hooks installed and the
slot invalid. Successful invalidation produces an empty slot after native zero
fills settle and calls the cache guard's reset callback exactly once.

## Model-local qualification

The CPU geometry fixture independently checks production layer counts, sizes,
frontiers and AR/MTP differences. The GPU adapter test compares every component
byte-for-byte at 2,047 and 8,203 tokens, covering unpooled warmup, incomplete
blocks and a wrapped physical ring cursor. Cross-slot continuations compare
exact logits, greedy tokens, seeded sampled rejection, actual draft counts,
acceptance, carried residuals and controller state. Edited continuations use the
same prefill pass boundaries: changing GEMM batch geometry is a separate
numerical contract, not a restore oracle. Additional cases cover empty/reset
state, guarded overwrite/release, missing bytes, submission failure, independent
capture streams during a peer's graph execution and image continuation.

`qwen38_flash_next_continuation_probe` is an explicitly built production tool.
It commits and touches 4 GiB of backing before filling every admitted live slot.
At 32k and 100k it measures private capture, full row materialization and a
cross-slot restore; it re-captures every piece and compares it exactly before
checking next-step logits and drafts. Its packed 64 MiB slabs measure the adapter
and physical backing envelope. They do not qualify card 19's checkpoint store
allocation policy: that policy must charge each actual backing assignment,
including slab slack, and select suitable precommitted block classes for the
256-row chunks and private component sizes.

Measured capacity and timings are recorded in [card 14 Results](../../cache-redesign/cards/14-flash-next-adapter.md#results-2026-10-10-draft).
The C=8 probe is a short-history row; eight long histories remain excluded.
Early HTTP integration of the new adapter requires card 19; no card 19 branch
is available yet. Existing HTTP qualification checks the inactive hooks only.
