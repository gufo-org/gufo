# Qwen 27B AR and DFlash2 continuation state

The adapter owns each target `hip::QwenGpuExecutor` and, in DFlash2 mode, its
draft backend. `QwenTextRunnerState` binds those component-cache slots and its
verifier borrows the slot's draft backend. AR and DFlash2 HTTP serving use this
path. The adapter adds no alternative model evaluator.

Each checkpoint contains the exact stable target frontier:

| Component | Kind | Representation |
| --- | --- | --- |
| Attention K and V, separately per attention layer | append rows | token/head/channel bytes; FP16 and FP32 have different layout versions |
| Convolution, per recurrent layer | private | all four FP32 taps, including the native saved tail |
| DeltaNet, per recurrent layer | private | FP32 production state or BF16 reference state, with distinct layout versions |
| Next-token logits | private | full FP32 vocabulary row; zero representation when no row has been computed |
| Position and generation frontier | private metadata | target position, logit validity, optional `GenerateFromPrefix` pending token |
| Image position/input identity | private metadata and attachment | SHA-256 of the computed image prefix; RoPE layout rebuilt from the matching request |
| Served generation frontier | private | selected token, publication flag, FP32 sampling logits and up to nine pending output IDs |
| DFlash2 K and V, separately per draft layer | private | FP32 physical ring bounded by context and sliding window; unused storage is zero |
| DFlash2 frontier and controller | private metadata | injected draft position, priming, pending feature count and adaptive controller state |
| DFlash2 pending target features | private | up to eight target feature rows in their original order, with zero padding |

For the retained 27B geometry, 16 attention layers contribute 64 KiB per token
with FP16 KV or 128 KiB with FP32 KV. Forty-eight recurrent layers contribute
48 × (163,840 convolution bytes + 3,145,728 DeltaNet bytes) in production.
The native vocabulary contributes 993,280 bytes and metadata contributes 64
bytes. Served frontier state adds another vocabulary row and a 96-byte header.
The AR private claim is 160,845,984 bytes. DFlash2 adds the bounded draft ring,
pending feature storage and a 64-byte header. The RFC's approximate 152 MiB
describes the target recurrent tensors.

FP16 native KV is token-major and can be borrowed without a capture copy.
FP32 native KV is head-major; transfers gather
or scatter each head with pitched copies without casting or rounding. Pieces
on independent nonblocking streams can arrive in any order. Private pieces
are bounded by each descriptor's state bytes, including tensor-crossing
offsets. Every load failure stays latched even if its completion is discarded.
Validation requires complete coverage and one exact frontier per component.

The caller supplies complete model, tokenizer and template identity. The
adapter adds state ABI, context, architecture geometry and the execution
policy fingerprint. Model artifacts, quantization, vision encoder and framing
must participate in the supplied identity. Distinct KV or recurrent storage
policies cannot share cache entries.

Mutation guards run synchronously on the host before decode, prefill chunks,
batched decode, batched verification, committed recurrent replay, reset,
legacy snapshot restoration and compact restoration. Every append range is
preserved before its first overwrite. Reset and restoration preserve all
borrowed rows while the old slot remains readable. Slot destruction drains
all transfers before the nonthrowing borrower release path. A failed native
mutation invalidates the slot until successful invalidation.

`SaveState`, replay capture and verification rollback are transaction-local
workspaces. A checkpoint cannot be captured between `SaveState` and
`FinishVerification`; those buffers are not an AR continuation frontier.
Stale saved recurrence and verification diagnostics are discarded on restore.
Ordinary forward passes overwrite activation, hidden, sampling and projection
scratch before reading it. Host prompt/verification feature captures are
request outputs, not AR resume state. DFlash2's pending target features are
retained private state. Its injected position can trail the target position;
private component index positions name the target checkpoint boundary, while
draft metadata carries the injected position independently. Captures also
refuse an outstanding draft proposal.

Images remain request attachments. Restoration requires the exact image
prefix identity and rejects missing or changed input. Text-only restoration
clears stale consumed image state, while a matching attached request retains
images after the checkpoint and their complete RoPE layout. Replacing images
already consumed by a live executor
requires an invalidation; image inputs after its frontier can be attached.
Legacy snapshots retain their existing caller responsibility to supply the
matching image input; their payload format is unchanged.

Serving uses committed host backing in equal segments of about 128 MiB, admitted
above both host and device heap reserves, and fixed 1 MiB disk staging. Each
prefill pass pins its entry recurrent buffer; writes use the existing rollback
pair. Cancellation can stop between layers and restore that pointer, preserving
completed prompt work without copying another checkpoint. The second recurrent
buffer is allocated before readiness and counted in the per-session memory
claim. Cancellation occurs before the target vocabulary head and draft
catch-up, leaving their previous frontier intact. Qwen's legacy server snapshot
hooks are removed; standalone native
snapshots remain available and DeepSeek keeps its legacy serving path.

Qwen restores use the destination executor's stream to avoid the pooled-stream
wait observed behind another session's cancelled prefill. Cold reassignment
preserves up to sixteen complete row chunks before waiting for their copies.
All copies and existing source readers settle before the old execution can be
reset; cancellation or a failed copy drains pending work before releasing its
source pins. Larger
ranges retain bounded piecewise preservation. Idle copying and disk staging
keep their existing bounds, and Flash-Next retains serial reassignment.

Earlier capacity and capture/restore measurements remain recorded in card 15
against their original revisions. Current Q4/Q8 AR/DFlash2 cache-serving checks
pass; the vision cold-prefill versus decoded-history discrepancy and broader
functional/resource gates remain open. Performance follows functional completion.

## Reproduce the adapter checks

Run the model-local check for each affected target GGUF. It uses assertions,
checks both KV layouts and loads the adjacent `mmproj-BF16.gguf` for images:

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test \
  --target qwen27b_continuation_adapter_test
nix develop -c build/gpu-test/tests/models/qwen27b/qwen27b_continuation_adapter_test "$MODEL"
```

`--guard-only`, `--image-only`, `--prefill-cancel` and `--draft "$DRAFT"` select
focused follow-ups. The cancellation check compares full component bytes and
resumed logits against an uninterrupted pass in both KV layouts.
The draft check restores into another slot and checks component bytes, injected
position, sampled proposals/RNG and feedback/controller state. The portable
`qwen27b_continuation_layout_test` covers checked geometry and policy identity;
it does not load model weights or establish numerical quality.

The step probe uses the production release configuration and requires two
native slots plus 8 GiB of committed checkpoint backing:

```sh
nix develop -c cmake --preset release -DGUFO_BUILD_TOOLS=ON
nix develop -c cmake --build --preset release --target qwen27b_continuation_probe
nix develop -c build/release/src/models/qwen/qwen27b_continuation_probe "$MODEL" 32768
nix develop -c build/release/src/models/qwen/qwen27b_continuation_probe "$MODEL" 100000
```

The retained qualification uses frozen candidate/main binaries and the same
benchmark recipes and functional harness. Its source pins, commands, raw
request measurements and comparison results are retained with the card's
evidence, outside Git.
