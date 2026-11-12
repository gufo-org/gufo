# Extension API

The extension API is the set of headers, CMake targets and manifest fields an
extension may use. This page explains how it is versioned and the contracts a
text provider implements; the headers under `include/gufo/ext/` document each
type.

## Versions

The API has its own version, `0.MINOR.REVISION`, separate from Gufo's. As of
November 2026 it is in the 0.2 series; `gufo diagnose --section extensions`
prints the exact version.

- A **revision** only adds things. An extension built for 0.2.10 builds against
  any later 0.2 revision.
- A **minor** bump may break extensions. The minor in your `api_version` must
  equal the host's, so `">= 0.1.12"` doesn't build against 0.2.
- Headers under `include/gufo/ext/experimental/` carry no promise; check them
  whenever you update.

The promise is source compatibility: rebuild against a new Gufo and your
extension compiles. Binaries built for one Gufo don't load into another.

Every change to the API adds a record under
[`include/gufo/ext/changes/`](../include/gufo/ext/changes/README.md), and the
revision is the number of records, so the records are the changelog.

## The boundary

Include only `<gufo/ext/...>` headers and link `gufo_ext_runtime`. Constants
generated from your manifest are under `<gufo/ext/generated/...>`. The build
rejects private Gufo headers and libraries for extensions outside the tree.

The `gufo` executable supplies the implementation when it links your
extension. A test or tool that is its own executable links the services it
uses:

| Target | Provides | Builds |
| --- | --- | --- |
| `gufo_ext_core` | GGUF reading, sampling, JSON constraints, dequantization, SHA-256, image decoding, hipBLASLt plans and, with HIP, `WeightUpload` | all |
| `gufo_ext_cache` | The component cache and, with HIP, its transfer and host-memory helpers | all |
| `gufo_ext_hip` | The HIP compile definitions and include paths the public HIP headers need | HIP only |

## Continuation

Every text runner supports continuation through the component cache. The cache
owns every execution slot and decides what to keep; the runner describes its
state and copies bytes when asked:

- `CreateComponentCacheResources` returns the adapter and its memory backing;
  build the backing with `hip::CreateHipComponentCacheResources`.
- `BindComponentState` wraps one slot in the runner's `TextRunnerState`.
- `ReconcileComponentState` runs after the cache restores a slot, so the runner
  can update its own bookkeeping to that position.
- `cache::Adapter` declares each component as **append rows**, final once
  written and shared between checkpoints, or **private state**, copied whole.
  Report positions per component, never from the prompt length.

`gufo/ext/transfer_pool.hpp` and `gufo/ext/snapshot_copy.hpp` help with the
copies. The cache identity must change whenever kept bytes stop being valid:
include the model files, the context size and every setting that changes state.
Test each component by restoring it and comparing byte for byte with an
uninterrupted run.

To capture checkpoints in the middle of a prefill pass, declare
`in_pass_checkpoint` and `max_in_pass_checkpoints` and implement
`PrefillCheckpointBytes` and `PrefillThrough`. Each capture must equal the state
a pass ending at that position leaves.

## Scheduling beside decoders

Every text runner declares a `TextSchedulingProfile` in
`TextRunnerDescriptor::scheduling` (header `gufo/ext/scheduling.hpp`); Gufo
refuses to load one that leaves a field at zero. While other sessions decode,
Gufo bounds each prompt pass with it so they don't wait for all of it:

- `pass_tokens` is the width Gufo uses for a prompt beside decoders, normally
  the model's own prefill pass. A prompt running alone isn't bounded.
- `max_layer_groups` is how many layer groups one such pass may be split into.
  One runs every pass unsplit, so decoders wait for the whole pass.
- `max_yield_decode_width` is how many sessions one decode step between groups
  may batch. Declare only what the loaded mode can run inside a yield.

Choose the groups from the time of a full pass:

- **Up to about half a second:** declare `max_layer_groups` 1. Decoders wait
  for one pass, which is no longer than a few decode steps.
- **Longer:** split the pass by layers. Decoders otherwise stall for the whole
  pass: 2 s on Qwen3.8-Flash-Next, 6.5 s on Qwen3.8-27B and 8 s on DeepSeek V4
  Flash at 4,096 tokens. Allow enough groups for about 250 ms each: the
  in-tree models declare 8 (Flash-Next) or 16.

To split, before such a pass Gufo calls `SetPrefillYield(groups, yield)` on the
prompt's state. `groups` comes from the measured time of the pass, capped at
`max_layer_groups` and at one group per 512 tokens.

- Split the pass into about that many groups of similar **duration**, not layer
  count, with a boundary right after any wait such as an input read. Between
  groups, call `state.YieldPrefill()`: it steps every other decoding session
  once, at most `max_yield_decode_width` per batch.
- Every layer must do the same arithmetic as an unsplit pass, so output stays
  bit-identical. While yielding, hold no lock or scratch the decode paths use.

A runner without `incremental_prefill` prefills every prompt whole in one
unsplit pass; its profile is checked but not used. In-tree tests check a
runner that splits passes with `gufo::test::CheckLayeredPrefill`
(`tests/support/layered_prefill_conformance.hpp`): every token and
log-probability must match between running alone and interleaved, on real
weights for each mode the profile covers.

## Sampling defaults

A provider recommends sampling through `TextRunnerDescriptor::sampling_defaults`
(the server) and `TextProvider::sampling_defaults` (`prompt` and `chat`): one
set of optional values each for thinking on, off and unspecified. Set both from
one function, so the server and the terminal agree. Values you set, zero
included, apply as written; unset controls stay neutral. Explicit server or
command-line settings override them, and a request overrides both.

## Reasoning markers

`TextModelRunner::ReasoningDelimiters()` returns the literal text that opens
and closes reasoning. The default follows `ReasoningBudgetTags()`; override it
when the model's text differs from its control tokens. `close` must not be
empty and `open` must not start with whitespace. With
`InitialOutputState::kThought` the prompt opened reasoning; with `kAuto` the
model's output does.

## Tool calls

A model whose call syntax isn't built in registers a `ToolDialect` and returns
its `id` from `TextModelRunner::ToolDialectId()`. The dialect drives both the
constrained grammar and the parsers of every chat endpoint, buffered and
streamed:

- `syntax` gives the call literals and either tagged key/value parameters or
  JSON-like arguments; Gufo compiles each tool's schema into it.
- `markers`, `reasoning_markers`, `closers` and `scan_tool_calls` let Gufo find
  calls in the text, including an unfinished call while streaming.

`ToolDialect` in [`text.hpp`](../include/gufo/ext/text.hpp) states the rules.
Runners that use a built-in format (`json`, `qwen`, `deepseek`) need nothing.

## Options

Options declared in the manifest appear in `--help` of `serve`, `prompt`,
`chat` and `bench` under "Provider options", parsed with their manifest types.
Two extensions may declare the same option if the types agree. Your provider
receives the values given on the command line in `provider_options`:

```cpp
std::uint32_t image_tokens = kDefaultImageTokens;
if (const auto* value = ext::FindOption<std::int64_t>(
        options.provider_options, "--image-tokens")) {
  image_tokens = ValidateImageTokens(*value);
}
```

An option that wasn't given is absent, so its default stays with you. Options
Gufo parses itself, such as `--mtp-model`, never appear here.

## Speculative defaults

`TextProvider::speculative_defaults` declares, per backend (`mtp`, `dflash2`,
`dspark`), the draft settings to use:

```cpp
.speculative_defaults = {{.backend = "mtp",
                          .max_draft_tokens = 15,
                          .min_draft_tokens = 1,
                          .draft_policy = "calibrated",
                          .draft_policies = {"calibrated", "fixed"}}},
```

Your values apply, else 7, 1 and no policy. `--draft-tokens`,
`--min-draft-tokens` and `--draft-policy` are deprecated and ignored.

## Files Gufo can't size

A GGUF whose tensors use a type or shape Gufo doesn't know, such as your own
quantization, still opens: those tensors have `sized == false`. Only a provider
that sets `accepts_unsized_tensors` is asked to probe such a file, and it reads
those tensors itself.

## NPU programs

`gufo/ext/experimental/npu.hpp` runs mlir-aie (IRON) designs on the Strix Halo
XDNA NPU through XRT. An **NPU program** is one xclbin and the instruction
streams built for it:

```cpp
namespace npu = gufo::npu;
npu::Device device = npu::Device::Open();
npu::Program prefill = npu::Program::Load(device, dir / "prefill.xclbin");
npu::Instructions insts = prefill.LoadInstructions(dir / "prefill.insts");
npu::Buffer weights = prefill.LoadBuffer(dir / "weights.bin", 4);
npu::Buffer state = prefill.Allocate(state_bytes, 3);
npu::Run run = prefill.Start(insts, {&state, &weights});
run.Wait();
state.FromDevice(0, logits_bytes);
```

Buffers are kernel arguments 3 and up and work in every program of the device.
`Start` returns at once, so you can sample while the NPU runs the next step.
Switching programs costs milliseconds; keep a prefill or a decode step in one.

For a design with run-time values (token, position), open its dispatch library
with `DispatchLibrary::Open` and generate streams. If only a few address words
change, `DiffInstructions` turns two generated streams into patches for
`Instructions::Apply`.

`npu::Q8Gemm` runs Q8_0 matrix products on the NPU for any row count, with
the weights resident in one arena. It loads the IRON design in
`src/core/npu/kernels/q8gemm`, which `q8gemm.py <dir>` compiles:

```cpp
npu::Q8Gemm gemm = npu::Q8Gemm::Open(device, dir);
int w = gemm.Upload(rows, cols, codes, scales);  // Q8_0 codes and block scales
gemm.Run(w, x, batch, y);                        // y [batch][rows] = x W^T
```

To compile your own designs, the dev shell has the IRON toolchain
(`GUFO_IRON_PYTHON`, `PEANO_INSTALL_DIR`); `gufo diagnose --section npu`
checks the device.

The header has no XRT types, so your extension builds without XRT headers;
Gufo links XRT itself. On a machine without an NPU, `Device::Open` throws
`npu::Error`. Call `ProbeDevices()` before you register an NPU path.
