# Qwen 27B autoregressive continuation state

The adapter owns a `hip::QwenGpuExecutor`, the same execution session held by
`QwenTextRunnerState`. Card 19 will attach the cache to that session. The legacy
runner continues to create its executor directly; every continuation hook is
null on that path. The adapter adds no alternative model evaluator.

Each checkpoint contains the exact stable target frontier:

| Component | Kind | Representation |
| --- | --- | --- |
| Attention K and V, separately per attention layer | append rows | token/head/channel bytes; FP16 and FP32 have different layout versions |
| Convolution, per recurrent layer | private | all four FP32 taps, including the native saved tail |
| DeltaNet, per recurrent layer | private | FP32 production state or BF16 reference state, with distinct layout versions |
| Next-token logits | private | full FP32 vocabulary row; zero representation when no row has been computed |
| Position and generation frontier | private metadata | target position, logit validity, optional `GenerateFromPrefix` pending token |
| Image position/input identity | private metadata and attachment | SHA-256 of the computed image prefix; RoPE layout rebuilt from the matching request |

For the retained 27B geometry, 16 attention layers contribute 64 KiB per token
with FP16 KV or 128 KiB with FP32 KV. Forty-eight recurrent layers contribute
48 × (163,840 convolution bytes + 3,145,728 DeltaNet bytes) in production.
The vocabulary contributes 993,280 bytes and metadata contributes 64 bytes.
The total production private claim is 159,852,608 bytes. It includes logits;
the RFC's approximate 152 MiB describes the recurrent tensors.

FP16 native KV is token-major. FP32 native KV is head-major; transfers gather
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
request outputs, not AR resume state. DFlash2's retained target features,
draft state and controller belong to card 16.

Images remain request attachments. Restoration requires the exact image
prefix identity and rejects missing or changed input. Text-only restoration
clears stale consumed image state, while a matching attached request retains
images after the checkpoint and their complete RoPE layout. Replacing images
already consumed by a live executor
requires an invalidation; image inputs after its frontier can be attached.
Legacy snapshots retain their existing caller responsibility to supply the
matching image input; their payload format is unchanged.

Production capacity and capture/restore measurements are recorded in card 15.
This adapter alone does not qualify the common cache allocator or the HTTP
switch-over. Those paths remain unattached until card 19.

## Reproduce the adapter checks

Run the model-local check for each affected target GGUF. It uses assertions,
checks both KV layouts and loads the adjacent `mmproj-BF16.gguf` for images:

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test \
  --target qwen27b_continuation_adapter_test
nix develop -c build/gpu-test/tests/models/qwen27b/qwen27b_continuation_adapter_test "$MODEL"
```

`--guard-only` and `--image-only` select focused follow-ups. The portable
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
