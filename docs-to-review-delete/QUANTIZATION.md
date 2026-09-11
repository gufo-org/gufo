# Quantization

Status: design draft, 2026-08-11

## Objective

Quantization is optimized for the selected model and Strix Halo hardware, not
for interchange with every inference runtime.

The source safetensors checkpoint is the quality oracle and conversion input,
not the intended serving representation. The quantizer searches for the best
quality available within declared model-size, memory, and performance targets.

The shared artifact family is provisionally named `SHQ-T16`. One artifact may
contain `SHQ4-T16`, `SHQ6-T16`, `SHQ8-T16`, and BF16 tensor encodings. The
current Python-produced layout is a candidate contract, not yet a frozen v1
artifact ABI. It combines established AWQ/GPTQ-style quantization with a
physical tile layout intended for consumption by gfx1151 and XDNA2 from one
resident weight copy.

These names describe deployment encodings, not new quantization algorithms.

## Terminology

Keep these concepts separate:

- Numerical quantization: codes, scales, zero points, and calibration policy.
- Physical layout: byte ordering, tiling, alignment, and metadata planes.
- Activation path: BF16, INT8, or another runtime activation representation.
- Device kernel: HIP or AIE2P implementation consuming the stored weights.

GPU and NPU should use the same numerical weights. They may use different
activation and accumulation paths.

## Release Objective

Each supported model has one release objective applied to every published size.
The objective defines:

- Absolute full-logit, perplexity, task, and routing quality floors. The
  task floors are capability-suite scores as defined in EVAL.md.
- Explicit encoded-byte targets for the selected size variants.
- Resident-memory budgets for each selected size.
- Protected single-request latency and throughput objectives.

The quantizer may use different precisions for sensitive tensors while
remaining within that model-wide objective and the selected artifact-size
budget.

A candidate for a selected size is promoted only when it is the best measured
overall tradeoff for that size and is on or improves the model's
quality-performance frontier. Evaluation includes:

- Full-logit KL, perplexity, task behavior (EVAL.md), and router agreement.
- Model size and resident memory.
- GPU decode latency.
- GPU and NPU prefill throughput.
- Heterogeneous execution performance.
- Conversion and runtime metadata overhead.

A smaller file is not automatically a better quantization. Unpacking,
additional scale traffic, restoration tensors, or slow device instructions can
make a lower-bit format slower end to end.

## Size Variants

The project may publish a few separately converted sizes for the same model.
They are size variants, not deployment profiles.

Every variant shares:

- The compiled `ModelKind` and model implementation.
- Tokenizer, chat template, and architecture revision.
- The `SHQ-T16` artifact family.
- CPU, HIP, and XDNA2 runtime support.

Every variant has its own:

- Weight shards and manifest.
- Quantization recipe and tensor-encoding table.
- Measured total bytes and effective bits per weight.
- Full-logit, perplexity, task, and performance reports.
- Artifact ID and checksums.

Names use measured quantities rather than subjective quality labels. For
example:

```text
Qwen3.8-27B-Text-SHQ-T16-4.4bpw
Qwen3.8-27B-Text-SHQ-T16-5.2bpw
Qwen3.8-27B-Text-SHQ-T16-6.1bpw
```

The displayed BPW is calculated from all encoded model tensor bytes divided by
the logical parameter count. Manifests also record the exact file and resident
sizes because metadata, alignment, tokenizer assets, and non-weight state are
not represented by BPW.

A server model alias selects one concrete artifact. Several sizes may be
configured under different aliases, but each consumes its own resident memory.
The runtime cannot reconstruct a higher-precision variant from a
lower-precision artifact.

The initial size range should be produced by changing the tensor-level mixture
of native Q4, Q6, Q8, and BF16 encodings. `SHQ6-T16` is a Q6_K-class storage
encoding (6.56 bpw): signed 6-bit codes packed 4-per-3-bytes, expanded to INT8
in the kernel. It exists because the unsloth Q4_K_M comparison showed the Q6_K
tier (embed/ffn_down/linear-attn) is exactly what closes the size gap without a
quality cliff (benchmarks/qwen3.5-0.8b/MIXED_PRECISION.md). Storage is not
compute: SHQ6 expands to native INT8 tiles before matrix multiplication, so
neither backend needs a packed Q6 matrix path. Packed Q5 is still avoided (no
natural Q5 matrix path; Q6 covers the intermediate tier).

Sub-four-bit variants require a separately specified Q2 or Q3 tensor encoding
and an efficient tile-local expansion path. They are later research and are not
part of the current `SHQ-T16` candidate contract.

## SHQ4-T16 Tensor Encoding

### Default numerical contract

```text
Weight codes:       UINT4
Scale:              BF16
Zero point:         UINT4
Default K group:    64
Quality K group:    32
Output tile:        16 channels
K microtile:        16 values
```

The asymmetric variant is named:

```text
SHQ4-T16-G64-U4Z
```

The symmetric variant stores two's-complement signed INT4 values and omits the
zero-point plane:

```text
SHQ4-T16-G64-S4
```

### Physical organization

Weights are output-major within a 16-channel tile, with adjacent K values
packed into each byte:

```text
qweight[n_tile][k_group][k16_subtile][output_lane=16][packed_k=8]
scales [n_tile][k_group][output_lane=16]                       BF16
zeros  [n_tile][k_group][output_lane=16]                       packed UINT4
```

One `K16 x N16` microtile contains 256 four-bit values and occupies 128 bytes.
The microtile is independently addressable and 128-byte aligned.

For G64:

```text
Four K16 weight microtiles: 512 bytes
Sixteen BF16 scales:         32 bytes
Sixteen UINT4 zero points:    8 bytes
Total:                      552 bytes for 1024 weights
Effective size:           4.3125 bits per weight
```

The symmetric G64 form is 4.25 bits per weight. G32 improves scale locality at
4.625 bits per weight for U4Z or 4.50 for S4.

Weights, scales, and zero points are stored as separate structure-of-arrays
planes. This keeps weight loads aligned and permits kernels to cache metadata
independently.

The byte layout is a design hypothesis until both backend microbenchmarks prove
that it is acceptably close to their device-native layouts.

## Candidate SHQ4-T16 Contract

The following rules define the candidate interoperable CPU, GPU, and NPU
format. They are normative for current experiments and conformance vectors, but
portable artifact compatibility is not frozen. The contract becomes
`SHQ4-T16 v1` only after the independent C++ parser, CPU decoder, HIP kernels,
promoted AIE programs, and malformed-artifact tests pass. Implementations may
use a lossless internal repack, but decoded values must agree throughout the
candidate-validation period.

### Logical tensor

The SHQ4 candidate stores a logical rank-2 linear weight:

```text
W[N, K]
N = output channels
K = input channels
```

The descriptor records logical and padded dimensions:

```text
N_padded = round_up(N, 16)
K_padded = round_up(K, group_size)
group_size = 32 or 64
```

Higher-rank source tensors are normalized by the model-specific importer before
quantization. The container records the logical role and any source
transformation; runtime kernels do not guess orientation.

### Byte and nibble order

- Multi-byte integers and BF16 values are little-endian.
- A weight byte stores the lower-K value in bits 0-3.
- The next K value is stored in bits 4-7.
- UINT4 codes are interpreted as `0..15`.
- S4 codes are two's-complement `-8..7`.

For one `K16 x N16` microtile:

```text
byte(output_lane, k_pair) =
    low_nibble(weight[output_lane, 2 * k_pair])
  | high_nibble(weight[output_lane, 2 * k_pair + 1])

output_lane = 0..15
k_pair = 0..7
```

The 128 microtile bytes are ordered first by `output_lane`, then by `k_pair`.

The byte offset in the weight plane is:

```text
offset =
  (((n_tile * k_group_count + k_group)
      * k16_per_group + k16_subtile)
      * 16 + output_lane)
      * 8 + k_pair
```

where:

```text
n_tile          = output_channel / 16
k_group         = input_channel / group_size
k16_per_group   = group_size / 16
k16_subtile     = (input_channel % group_size) / 16
```

### U4Z decoding

Each real output channel and K group stores one BF16 scale `s` and one UINT4
zero point `z`:

```text
dequant(q) = float(s) * (int(q) - int(z))
q in [0, 15]
z in [0, 15]
```

Scale entries are ordered:

```text
scale_index = (n_tile * k_group_count + k_group) * 16 + output_lane
```

Zero points use the same logical index. Two adjacent output-lane zero points
are packed per byte, with the even lane in the low nibble.

For a nonzero group, `s` must be positive and finite. The quantizer computes
candidate parameters in FP32 and rounds the selected scale to BF16 using
round-to-nearest, ties-to-even before final code selection.

The canonical all-zero group is:

```text
s = BF16 zero
z = 0
all q = 0
```

### S4 decoding

S4 has no zero-point plane:

```text
signed_q = nibble < 8 ? nibble : nibble - 16
dequant(q) = float(s) * signed_q
```

All signed codes `-8..7` are legal. The canonical all-zero group has a zero
scale and zero codes.

### Quantizer rounding

Unless a versioned recipe explicitly declares another deterministic search:

```text
q_real = w / s + z       U4Z
q_real = w / s           S4
q = clamp(round_to_nearest_ties_to_even(q_real), code_min, code_max)
```

Imatrix weighting changes parameter selection, not code interpretation.

NaN or infinite source weights are rejected. They are not silently clamped or
encoded.

### Padding

- Padded U4Z K values store `q = z`, which dequantizes to zero.
- Padded S4 K values store code zero.
- Padded output channels store zero scales and zero codes.
- Padding is included in file checksums but excluded from quality statistics.
- Kernels must mask logical output tails before publishing results.

### Dynamic A8 activation contract

The shared NPU and heterogeneous prefill path uses symmetric dynamic INT8
activations grouped over the same K groups as the weights.

For row `r` and K group `g`:

```text
amax = max(abs(x[r, g]))
a_scale = amax / 127
qa = clamp(round_to_nearest_ties_to_even(x / a_scale), -127, 127)
```

If `amax` is zero, `a_scale` and all codes are zero. Canonical activation
quantization does not emit `-128`.

Store one BF16 activation scale per row and K group. BF16 conversion uses
round-to-nearest, ties-to-even. A split GPU/NPU operator consumes one shared
activation code buffer and the same BF16 scale plane.

Another scale dtype requires a distinct activation numerical-contract ID and
cannot participate in the canonical split route.

### W4A8 accumulation

For U4Z:

```text
dot[r, n, g] =
    sum_k int32(qa[r, k]) * int32(qw[n, k])

corrected[r, n, g] =
    dot[r, n, g]
  - int32(z[n, g]) * sum_k int32(qa[r, k])

partial[r, n, g] =
    float(corrected[r, n, g])
  * float(a_scale[r, g])
  * float(w_scale[n, g])
```

For S4, omit the zero-point correction.

INT32 multiplication and group sums are exact. Group partials are accumulated
in FP32. A kernel that reassociates group accumulation must declare a distinct
numerical-contract ID and pass its logit-quality gate.

Bias and activation functions are model-operation epilogues and are not part of
the SHQ4-T16 weight encoding.

### BF16 activation path

GPU decode may consume BF16 activations without A8 conversion:

```text
sum_k float(x[k]) * dequant(qw[k])
```

The kernel accumulates in FP32 and declares its reduction-order contract. This
path uses the same stored weight codes, scales, and zero points but is not
expected to be bit-identical to W4A8.

### Format identifiers

Examples:

```text
SHQ4_T16_V1_U4Z_G32
SHQ4_T16_V1_U4Z_G64
SHQ4_T16_V1_S4_G32
SHQ4_T16_V1_S4_G64
```

The tensor descriptor also records:

- Logical and padded shape.
- Weight, scale, and zero-plane offsets and lengths.
- Alignment of every plane.
- Stored scale dtype.
- Weight and activation numerical-contract IDs.
- Payload checksum.

Unknown versions or identifiers fail closed.

### Conformance vectors

The repository must contain tiny byte-exact vectors for:

- Every UINT4 and signed nibble value.
- Low/high nibble ordering.
- G32 and G64 offsets.
- U4Z zero correction.
- S4 negative values including `-8`.
- BF16 scale rounding.
- Zero groups and padded K/N tails.
- Dynamic A8 rounding and saturation.
- Exact INT32 group accumulation.
- FP32 epilogue reference outputs.

CPU, HIP, and AIE implementations must all consume the same vectors.

## SHQ8-T16 Tensor Encoding

SHQ8-T16 is the native higher-precision integer encoding used for tensors that
do not meet the release quality floor in SHQ4-T16.

The initial numerical contract is:

```text
Weight codes:       signed INT8
Scale:              BF16
Zero point:         none
Default K group:    64
Output tile:        16 channels
K microtile:        16 values
```

Decoding is:

```text
dequant(q) = float(scale) * int(q)
q in [-127, 127]
```

The canonical quantizer does not emit `-128`. Each output channel and K group
has one positive BF16 scale, except an all-zero group, which has a zero scale
and zero codes.

One `K16 x N16` microtile contains 256 signed bytes and occupies 256 bytes.
Four microtiles plus sixteen BF16 scales form a G64 tile:

```text
Four K16 weight microtiles: 1024 bytes
Sixteen BF16 scales:          32 bytes
Total:                      1056 bytes for 1024 weights
Effective size:             8.25 bits per weight
```

The output-lane, K-subtile, padding, BF16 rounding, and alignment conventions
match SHQ4-T16. Complete candidate SHQ8 byte-offset and conformance-vector
sections must be added before its kernels are promoted or the family is frozen
as v1.

## SHQ6-T16 Tensor Encoding

SHQ6-T16 is the Q6_K-class storage encoding (6.56 bpw) for tensors that need
more than SHQ4 but less than SHQ8 (embed, ffn_down, linear-attn projections).
It closes the unsloth Q4_K_M size gap (see MIXED_PRECISION.md). Storage is not
compute: SHQ6 expands to native INT8 tiles before matrix multiplication.

```text
Weight codes:       signed INT6 (-32..31)
Scale:              BF16
Zero point:         none
Default K group:    64
Output tile:        16 channels
K microtile:        16 values
```

Decoding is:

```text
dequant(q) = float(scale) * signed6(q)
signed6: code < 32 ? code : code - 64
```

Packing is 4 codes per 3 bytes, little-endian 24-bit stream along K:

```text
byte0 = v0 | ((v1 & 3) << 6)
byte1 = (v1 >> 2) | ((v2 & 15) << 4)
byte2 = (v2 >> 4) | (v3 << 2)
```

One `K16 x N16` microtile contains 256 6-bit codes = 192 bytes. Four
microtiles plus sixteen BF16 scales form a G64 tile:

```text
Four K16 weight microtiles: 768 bytes
Sixteen BF16 scales:         32 bytes
Total:                       800 bytes for 1024 weights
Effective size:            6.5625 bits per weight
```

A complete candidate SHQ6 byte-offset, validation, and conformance-vector
section is required before its kernels are promoted or the family is frozen as
v1.

Output-lane, K-subtile, padding, BF16 rounding, and alignment conventions
match SHQ8/SHQ4. Conformance vectors cover the 4-per-3-byte bit packing and
signed 6-bit round-trip.

## Backend Arithmetic

### XDNA2

The primary NPU path is:

```text
dynamic INT8 activation x UINT4 weight -> INT32 group accumulator
INT32 -> FP32/BF16 scale epilogue
```

For U4Z:

```text
sum(a * (q - z)) = sum(a * q) - z * sum(a)
```

The activation sum is computed once per row and K group, then reused across 16
output channels.

### gfx1151

GPU decode may use:

```text
BF16 activation x dequantized UINT4 weight -> FP32 accumulation
```

This avoids dynamic activation quantization for single-token GEMV.

GPU prefill and batched verification may use INT8 activations or stage the
weights into BF16/FP16 WMMA tiles. The stored weights remain unchanged.

When GPU and NPU split one operator, both must consume the same shared INT8
activation representation for that operator unless backend equivalence tests
approve another contract.

## Imatrix and Activation Calibration

Imatrix support is mandatory in the quantization toolchain.

The baseline objective is activation-weighted reconstruction error:

```text
weighted_error = sum_i importance[i] * (w[i] - dequant(q[i]))^2
importance[i] ~= E[x_i^2]
```

Support:

- Import of llama.cpp-compatible importance matrices where dimensions match.
- Native calibration artifacts with model, tensor, tokenizer, dataset, and
  sequence-length hashes.
- Per-input-channel second moments.
- Activation maxima, percentiles, and outlier frequency for NPU INT8 paths.
- Per-layer output samples for reconstruction testing.

Imatrix data is used during conversion and mixed-precision search. It is not
required by the serving runtime.

Calibration and held-out quality evaluation must use disjoint prompt sets.
Quantizer search must not tune against the release suite.

## Offline Toolchain

Serving remains C++20 and torch-free. Offline conversion may use Python,
PyTorch, Transformers, and safetensors when they provide a reliable
full-quality reference or activation-capture path.

The long-term tool split is:

- C++20 for safetensors inspection, deterministic quantization, packing,
  checksums, container writing, and CPU reference kernels.
- Python for model-specific reference loading, activation capture, dataset
  orchestration, recipe search, report generation, and Hugging Face Hub access.
- Shared JSON schemas at the boundary so either side can be replaced without
  changing the artifact contract.

Offline Python dependencies must never become transitive dependencies of the
server or compiled-in model implementations.

## Safetensors Source Contract

### Accepted source

The converter accepts either:

- A local Hugging Face model snapshot directory.
- A Hugging Face model repository plus an immutable commit revision.

Branch names such as `main` may be accepted for interactive inspection, but a
conversion cannot be published until the branch is resolved to a commit hash.

The source directory normally contains:

```text
config.json
generation_config.json                  optional
tokenizer.json                          or model-specific tokenizer files
tokenizer_config.json
special_tokens_map.json                 optional
chat_template.jinja                     optional
model.safetensors                       single-file checkpoint
```

or:

```text
model.safetensors.index.json
model-00001-of-000NN.safetensors
model-00002-of-000NN.safetensors
...
```

The loader reads `weight_map` from the index as structured JSON. It must not
infer shard ownership from filenames or lexical ordering.

Pickle-based `.bin` or `.pt` checkpoints are rejected by default. An explicit
research-only import step may convert them to safetensors before the normal
pipeline, but publication provenance must identify that conversion.

### Validation

Before reading tensor payloads:

- Validate safetensors header length, JSON, dtype, shape, byte offsets, and
  non-overlap.
- Reject duplicate tensor names and missing indexed tensors.
- Reject files not referenced by the selected index unless explicitly allowed.
- Verify tensor byte length from dtype and shape.
- Hash every source file with SHA-256.
- Record source repository, resolved revision, and upstream license metadata.
- Verify that the tokenizer and model configuration describe a supported model
  package.

The converter reads tensors lazily by memory mapping or bounded buffered I/O.
It must not require the entire full-quality checkpoint and the entire quantized
checkpoint to be resident simultaneously.

### Source manifest

Inspection creates `source-manifest.json`:

```json
{
  "schema": "gufo.source.v1",
  "repository": "organization/model",
  "revision": "<immutable-commit>",
  "architecture": "<supported-model-kind>",
  "source_storage_dtype": "BF16",
  "tokenizer_sha256": "<sha256>",
  "chat_template_sha256": "<sha256>",
  "files": [
    {
      "path": "model-00001-of-00012.safetensors",
      "size": 123,
      "sha256": "<sha256>"
    }
  ]
}
```

All later artifacts refer to this manifest hash.

## Model-Specific Import Adapters

Each supported architecture owns an import adapter. The adapter maps Hugging
Face tensor names and configuration fields to stable logical roles:

```text
embedding
layer.<n>.attention.q_proj
layer.<n>.attention.k_proj
layer.<n>.attention.v_proj
layer.<n>.attention.o_proj
layer.<n>.ffn.gate
layer.<n>.ffn.up
layer.<n>.ffn.down
router
expert.<n>.gate_up
expert.<n>.down
final_norm
lm_head
mtp_or_support_head
```

The adapter must explicitly handle:

- Tied embeddings and output weights.
- Transposed or fused source tensors.
- Grouped-query and latent-attention metadata.
- MoE expert ordering and router semantics.
- Shared experts.
- MTP, DSpark, or other support-model weights.
- Rope parameters and model-specific recurrent state.

Generic name guessing is not accepted for a released model. Import tests use a
complete expected tensor inventory and fail on missing, extra, or ambiguous
tensors.

## End-to-End Quantization Pipeline

### 1. Resolve and inspect

Resolve the source revision, acquire the required files, validate safetensors,
and write `source-manifest.json`. Inspection reports tensor count, source dtype,
logical role, dimensions, byte size, tied storage, and proposed quantization
eligibility.

No quantization starts while the tensor inventory is incomplete.

### 2. Establish the teacher

Run the source checkpoint through a pinned full-quality reference
implementation. The teacher uses the source storage type and records its
accumulation settings.

Capture:

- Full-vocabulary logits on the fixed teacher-forced suite.
- Perplexity/NLL corpus results.
- Selected layer-boundary tensors.
- MoE router logits and chosen experts where applicable.

The reference may run on another machine. Its artifacts remain valid only when
the source, tokenizer, template, prompt suite, and reference-runtime hashes
match.

### 3. Calibrate

Run representative calibration prompts through the full-quality model and
collect:

- Per-input-channel `E[x^2]` imatrix values.
- Activation maxima and selected percentiles.
- Signed ranges and outlier frequency.
- Representative layer inputs and outputs.
- Router inputs and expert-selection frequency.
- Context-length and modality category metadata.

Support both native calibration artifacts and compatible llama.cpp imatrix
imports. Imported data is accepted only when tensor dimensions and the source
model identity match.

#### Calibration compute (gufo-calibrate)

The imatrix reduction `h[j] = E[x_j^2]` is always accumulated in float64. The
corpus is forwarded in token-budgeted batches (`--max-tokens`, default 4096);
pad positions are excluded with the attention mask, never summed, so batching
does not change per-real-token mathematics. Runs are bit-reproducible for a
fixed `--device` and batch budget. `--device cuda` offloads the forward to the
gfx1151 GPU (ROCm torch; unified default shell) for multi-million-token
corpora; the GPU path forces fp32 accumulation (`allow_tf32=False`) and
deterministic algorithms. `--max-tokens 1` reproduces the legacy one-prompt
per-forward result; `--reference DIR` cross-checks a new artifact against a
prior one.

Batch size and device change the forward's low bits (GEMM accumulation order,
as with any parallel matmul), so activations agree only to fp32 precision and
the difference propagates through the 24+ layers; on low-energy channels this
is larger in relative terms. This is not pad leakage (the first projected layer
matches bit-for-bit across batch sizes) and it does not affect the objective:
quantized reconstruction rmse shifts by <0.03% between sequential, batched,
and GPU calibration on the 0.8B reference.

Raw private calibration text should not be embedded in a published model.
Publish dataset identifiers, revisions, licenses, category counts, and hashes
when redistribution of the raw samples is not appropriate.

### 4. Search a recipe

For each eligible tensor:

1. Establish unweighted S4 and U4Z G32/G64 candidates.
2. Optimize scales and zero points using imatrix-weighted error.
3. Evaluate optional AWQ or SmoothQuant channel equalization.
4. Evaluate GPTQ-style reconstruction against representative layer inputs.
5. Measure tensor reconstruction and layer-output error.
6. Promote sensitive tensors or channel ranges to G32, Q8, or BF16.

Search is constrained by:

- Target model bytes.
- Peak conversion memory.
- GPU and NPU kernel support.
- Decode and prefill latency.
- Full-model quality budgets.

The result is `quantization-plan.json`. It records the chosen format and the
rejected alternatives for every tensor. The plan is deterministic and can be
reviewed before conversion.

### 5. Quantize and pack

Process one bounded tensor region at a time:

```text
read source slice
-> normalize logical orientation
-> apply approved equalization
-> quantize with pinned algorithm
-> pack into SHQ T16 planes
-> run block oracle
-> checksum output
-> release source pages
```

Large tensors may be processed in output-channel stripes, but group boundaries
must never be split incorrectly. The final bytes must be invariant to worker
count, shard size, and scheduling order.

Write to temporary shard names and atomically finalize a shard only after its
index, payload, and checksum validate. A conversion can resume by verifying and
skipping completed shards.

### 6. Validate the converted model

Validation proceeds from narrow to broad:

- Pack/dequantize CPU fixtures.
- Every converted tensor against its recorded reconstruction budget.
- Layer boundaries against the full-quality teacher.
- GPU and NPU kernels against CPU reference operations.
- Full-model teacher-forced logits.
- Perplexity and capability suites.
- GPU-only, NPU-only, and heterogeneous execution routes.

The requirements and artifact format are defined in
[Testing and regression control](TESTING.md). A failed quality gate prevents
packaging and publication even when the model loads and generates text.

### 7. Package

The proposed Hugging Face artifact directory is:

```text
README.md
LICENSE or upstream license files
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
chat_template.jinja
gufo-manifest.json
source-manifest.json
quantization.json
quantization-plan.json
model-00001-of-000NN.gufo
model-00002-of-000NN.gufo
model.gufo.index.json
checksums.sha256
evals/quality.json
evals/performance-gfx1151-xdna2.json
```

Only files present in the source are copied from the optional tokenizer and
generation-config list. The package must not invent tokenizer defaults.

`gufo-manifest.json` is the runtime entry point. It references immutable shard
checksums, model kind, model-state contract version, layout version, required
`gfx1151` and XDNA2 capabilities, quantization recipe, and tokenizer identity.
It cannot name or supply executable host or device code.

### 8. Publish

Publication is an explicit release action, never an automatic side effect of
conversion.

The publisher:

1. Verifies redistribution rights and preserves upstream attribution.
2. Runs all release gates and validates the local package checksums.
3. Generates or validates the Hugging Face model card.
4. Creates a private staging repository or staging revision.
5. Uploads the complete artifact directory.
6. Downloads or remotely inspects the uploaded revision and verifies hashes.
7. Records the resulting Hub commit in the local release report.
8. Makes the repository public or promotes the verified revision only after
   explicit approval.

Use the current Hugging Face `hf upload` command or
`HfApi.upload_folder()`. These paths support streamed, resumable folder uploads.
The older `upload_large_folder()` and `hf upload-large-folder` interfaces are
deprecated and must not be built into the tool.

Authentication uses `hf auth login` or `HF_TOKEN`. Tokens must not appear in
commands retained in reports, process arguments emitted to logs, model
metadata, or shell history.

For maximum upload performance, the publisher may expose the current
`HF_XET_HIGH_PERFORMANCE=1` option, but correctness and resumability must not
depend on it.

## Proposed Conversion Utilities

The first tool surface should be:

| Utility | Responsibility |
| --- | --- |
| `gufo-inspect` | Resolve and validate source safetensors and write source inventory |
| `gufo-calibrate` | Capture imatrix, activation ranges, layer samples, and router data |
| `gufo-plan-quant` | Search mixed-precision choices and write a reviewable plan |
| `gufo-quantize` | Deterministically convert and pack resumable output shards |
| `gufo-validate` | Run tensor, layer, logit, perplexity, and task gates |
| `gufo-package` | Assemble manifests, tokenizer assets, reports, and model card |
| `gufo-upload` | Stage, upload, verify, and optionally publish a Hub artifact |

These are proposed interfaces for implementation, not commands that already
exist.

Example flow:

```bash
gufo-inspect \
  --source organization/model \
  --revision <immutable-commit> \
  --out work/source-manifest.json

gufo-calibrate \
  --source-manifest work/source-manifest.json \
  --suite calibration/suite.json \
  --out work/calibration/

gufo-plan-quant \
  --source-manifest work/source-manifest.json \
  --calibration work/calibration/manifest.json \
  --target-bytes 18000000000 \
  --out work/quantization-plan.json

gufo-quantize \
  --source-manifest work/source-manifest.json \
  --plan work/quantization-plan.json \
  --out work/model/

gufo-validate \
  --teacher work/teacher/manifest.json \
  --candidate work/model/gufo-manifest.json \
  --suite quality/release-suite.json \
  --out work/validation/

gufo-package \
  --model work/model/ \
  --validation work/validation/ \
  --out release/model-shq-t16-4p4bpw/

gufo-upload \
  --directory release/model-shq-t16-4p4bpw/ \
  --repo-id organization/model-SHQ-T16-4p4bpw-gufo \
  --private \
  --dry-run

gufo-upload \
  --directory release/model-shq-t16-4p4bpw/ \
  --repo-id organization/model-SHQ-T16-4p4bpw-gufo \
  --private \
  --execute
```

`--dry-run` is a Gufo wrapper feature. It validates the file inventory,
license, sizes, checksums, authentication state, destination, and release
report without uploading.

`--execute` performs the upload. Repository visibility remains controlled by
`--private` or an explicit later promotion command; uploading must not make a
private staging repository public implicitly.

Every utility supports:

- `--json` machine-readable output.
- A nonzero exit code on validation failure.
- A work directory containing resumable state.
- Immutable input hashes in its output manifest.
- `--threads` without changing produced bytes.
- Refusal to overwrite a different artifact without `--force`.

## Hugging Face Model Card

The package `README.md` must clearly state that the weights are a Gufo Engine
deployment format and are not standard Transformers safetensors.

Recommended metadata includes:

```yaml
---
library_name: gufo
pipeline_tag: text-generation
base_model: organization/source-model
base_model_relation: quantized
license: <upstream-license-id>
tags:
  - gufo
  - gfx1151
  - xdna2
  - quantized
---
```

The model card also records:

- Exact source repository and commit.
- Quantization recipe, BPW, group sizes, and mixed-precision policy.
- Calibration dataset identifiers and revisions where publishable.
- Imatrix and recipe hashes.
- Full-quality teacher dtype and reference runtime.
- Logit, KL, perplexity, task, and performance reports.
- Required Gufo Engine version and model kind.
- Supported context and KV-cache formats.
- Known limitations.
- Upstream license, attribution, and use restrictions.

Hugging Face documents model cards as repository `README.md` files with YAML
metadata, including explicit `library_name`, `base_model`, license, and
evaluation information. The publisher validates these fields before upload.

## Mixed Precision

Do not force every tensor into Q4. An `SHQ-T16` artifact may include:

| Format | Intended use |
| --- | --- |
| `SHQ4-T16-G64-U4Z` | Bulk linear tensors |
| `SHQ4-T16-G32-U4Z` | Sensitive attention or output tensors (see note) |
| `SHQ4-T16-G64-S4` | Fast symmetric tensors |
| `SHQ6-T16-G64` | Q6_K-class tier: embed, ffn_down, linear-attn |
| `SHQ8-T16-G64` | Difficult tensors and sensitive experts |
| `BF16` | Norms, routers, selected heads, reference paths |

For text-only serving, the source vision encoder (`model.visual.*`, ~200 MB
bf16 on Qwen3.5-0.8B) should be dropped from the artifact, matching llama.cpp /
unsloth. It is dead weight for text inference and is the dominant size lever.

Evidence (benchmarks/qwen3.5-0.8b/MIXED_PRECISION.md, Qwen3.5-0.8B):

- ffn_down -> SHQ8 is the top quality lever (matched-token KL 0.1154 -> 0.0862).
- embed -> SHQ8 is a free size cut: quality-neutral, 246 MB smaller. The
  BF16-embed policy over-spends the largest tensor; embeddings do not need
  full precision when the LM head is tied to them.
- SHQ6 (6.56 bpw) replaces SHQ8 on the upcast set: saves 86-134 MB at a small
  quality cost (shq6_ffn 0.0886 vs 0.0866; shq6_mirror 0.0498 vs 0.0383). It is
  the Q6_K-class tier that closes the unsloth size gap.
- SHQ4-G32 attention measured no quality gain on the 78-position suite; treat
  G32 as optional, not default (it adds a second group-size kernel path).
- linear_attn projections -> SHQ8/SHQ6 is the biggest further lever (KL 0.0866
  -> 0.0383 / 0.0498), mirroring unsloth Q8_0/Q5_K linear-attn choices.
- Vision tower (~200 MB bf16) should be dropped for text-only serving; llama.cpp
  / unsloth drop it. Dominant size lever, independent of quantization.

SHQ6/SHQ8 share the SHQ4 T16 tile layout and packing order (byte-exact), so
they are one kernel family, not new ones; only the per-weight read width
differs (SHQ6: 3 bytes per 4 weights, SHQ8: 1 byte per weight). SHQ6 expands to
INT8 in the kernel. Keep the upcast tier small on the decode GEMV path (decode
is bandwidth-bound; SHQ8 costs 2x bytes/weight, SHQ6 ~1.46x). Precision
selection is per-tensor, never per-block, so hot kernels do not branch per block.

Precision selection should be at tensor or contiguous channel-range
granularity. Per-block format branching is avoided in hot kernels.

For MoE models, separately evaluate:

- Router logits and top-k expert agreement.
- Shared experts.
- Routed expert gate/up and down projections.
- Indexer or compressor tensors.
- Output and speculative-support heads.

An experimental two-bit S40 storage format may later be evaluated for routed
expert bulk. It must expand into native INT4 tiles before matrix multiplication
and is not part of the first implementation.

## Quality Evaluation

The full-quality checkpoint is the reference. All intrinsic comparisons use
matched-token teacher forcing as defined in
[Testing and regression control](TESTING.md). Required measurements include:

- Perplexity and NLL.
- Mean, p99, p99.9, and maximum full-vocabulary KL divergence.
- Top-1, top-5, and top-k token agreement.
- Per-layer cosine similarity and normalized RMS error.
- Attention Q/K score error.
- MoE router top-k and selected-expert agreement.
- Tool-call syntax and semantic success.
- JSON schema validity.
- Code execution tests.
- Long-context retrieval and continuation.
- Speculative proposal acceptance and target commit agreement.

Quality promotion thresholds are model- and artifact-specific. A lower
perplexity on a small corpus does not override severe tail-KL or task failures.

Performance is evaluated for the same artifact and held-out workload. Release
reports place candidates on quality-versus-latency, quality-versus-throughput,
and quality-versus-memory curves. The selected artifact must satisfy its
absolute quality floor and improve the model's complete serving objective.

Free-running generated outputs are evaluated separately. After the first token
divergence, their later logits are not treated as a direct measure of
quantization error.

## Container Metadata

Each tensor records:

- Logical dtype and physical layout version.
- Shape and padded shape.
- Group size and output tile.
- Scale and zero-point plane offsets.
- Required alignment.
- Quantization recipe ID.
- Calibration artifact hash.
- Full-quality source checkpoint hash.
- Tensor checksum.
- Allowed backend kernel variants.

The model manifest records:

- Model architecture and exact revision.
- Tokenizer and chat-template revision.
- Artifact family `SHQ-T16`.
- Exact artifact ID, total encoded bytes, resident bytes, and measured BPW.
- Quantization recipe ID.
- Tensor counts and bytes by `SHQ4-T16`, `SHQ8-T16`, and BF16 encoding.
- Supported GPU and NPU architecture IDs.
- Quality report location and promotion status.

Unknown layout versions must fail closed.

## Tests

- Exhaustive UINT4 and signed nibble decode.
- Exhaustive SHQ8 signed-byte decode, rounding, saturation, and offsets.
- Scale and zero-point edge cases.
- Exact INT32 accumulation against a wide CPU accumulator.
- Padding and tail dimensions.
- T16 address and alignment tests.
- Weighted quantizer determinism.
- Imported imatrix compatibility.
- CPU dequantized layer oracle.
- GPU versus CPU and NPU versus CPU.
- GPU/NPU split operator versus unsplit operator.
- Full-model greedy token and logit tests.
- Reproducibility and independent quality reports for every published size.
- Manifest rejection when the measured BPW, byte inventory, or encoding table
  disagrees with the payload.
- Corruption and checksum rejection.

## External References

- Safetensors documentation:
  `https://huggingface.co/docs/safetensors/index`
- Hugging Face Hub upload documentation:
  `https://huggingface.co/docs/huggingface_hub/guides/upload`
- Hugging Face CLI documentation:
  `https://huggingface.co/docs/huggingface_hub/guides/cli`
- Hugging Face model-card documentation:
  `https://huggingface.co/docs/hub/model-cards`
