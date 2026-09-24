# Qwen implementation

The dense Qwen3.8 27B runtime owns target inference, native MTP and DFlash2.
Shared Qwen tokenization, chat templates, vision and sampling are also used by
Flash-Next. Audio implementations are model-private.

[27B usage](../../../docs/models/qwen3.8-27b/README.md) ·
[27B quality](../../../docs/models/qwen3.8-27b/QUALITY.md) ·
[Flash-Next](../../../docs/models/qwen3.8-flash-next/README.md)

## Source ownership

| Path | Responsibility |
|---|---|
| [`state.hpp`](state.hpp), [`state.cpp`](state.cpp) | Non-owning tensor references plus CPU KV cache and activation scratch state. |
| [`weights.cpp`](weights.cpp) | GGUF tensor-name binding, role-specific format/shape checks, encoded-storage validation, and mixed-layer-kind validation. |
| [`forward.hpp`](forward.hpp), [`forward.cpp`](forward.cpp) | CPU/reference tensor math, layer composition, full single-token forward, and sampling oracle. |
| [`generator.hpp`](generator.hpp), [`generator.cpp`](generator.cpp) | CPU autoregressive generation loop and GGUF-backed reference generator. |
| [`ssm.hpp`](ssm.hpp), [`ssm.cpp`](ssm.cpp) | CPU Gated DeltaNet cache and the typed SSM parameter-slice implementation. |
| [`modules/`](modules/) | Narrow stage contracts for embedding, norm, attention, RoPE, residual, FFN, SSM, unembedding, sampling, and quantized GEMM. [`module_ctx.hpp`](modules/module_ctx.hpp) separates CPU and HIP capabilities. |
| [`gemm_route.hpp`](gemm_route.hpp) | Pure, host-only format/shape/capability-to-GEMM-route resolver shared by CPU, HIP decode, HIP prefill, and HIP MTP. |
| [`hip/executor.hpp`](hip/executor.hpp), [`hip/executor.cpp`](hip/executor.cpp) | Immutable GPU-visible model, session executor, arena-facing API, generation loop, graph identity, and persistent execution state. |
| [`hip/decode.cpp`](hip/decode.cpp), [`hip/decode_step.cpp`](hip/decode_step.cpp) | Token I/O and graph orchestration versus the per-token decode composition root. |
| [`hip/prefill.cpp`](hip/prefill.cpp), [`hip/prefill_chunk.cpp`](hip/prefill_chunk.cpp) | Prompt chunking versus the batched prefill launch chain. |
| [`hip/arena.cpp`](hip/arena.cpp), [`hip/ssm_replay.cpp`](hip/ssm_replay.cpp) | Stable GPU allocation ownership versus recurrent snapshots and SSM replay. |
| [`hip/execution_policy.hpp`](hip/execution_policy.hpp) | Immutable execution policy, pure per-layer route resolution, rejection reasons, fingerprints, and graph eligibility. |
| [`hip/ops/`](hip/ops/) | Narrow launcher interfaces grouped by attention, GEMM, norm/residual, SSM, SwiGLU, and token operations. [`ops.hpp`](hip/ops.hpp) is a compatibility umbrella. |
| [`hip/kernels/`](hip/kernels/) | HIP implementations split by launch/experiment boundary; graph-pointer attention, decode recurrence, quant GEMV, and fused RMSNorm+SwiGLU have independent translation units. |
| [`mtp_reference.hpp`](mtp_reference.hpp), [`mtp_reference.cpp`](mtp_reference.cpp) | Stateful CPU oracle for the single-layer MTP graph. |
| [`hip/mtp/`](hip/mtp/), [`hip/mtp.hpp`](hip/mtp.hpp) | GPU MTP model conversion, executor, and speculative draft backend. |
| [`dflash_weights.hpp`](dflash_weights.hpp), [`dflash_weights.cpp`](dflash_weights.cpp) | DFlash2 GGUF configuration, tensor binding and validation. The independent operator reference is in `tools/qwen27b/dflash_reference.py`. |
| [`hip/dflash/`](hip/dflash/), [`hip/dflash.hpp`](hip/dflash.hpp), [`hip/kernels/dflash_kernels.*`](hip/kernels/dflash_kernels.hip) | GPU DFlash / DFlash-2 model, non-causal block attention kernels, 2-tap dynamic convs, bilinear path selector, and speculative draft backend. |
| [`tokenizer.*`](tokenizer.hpp), [`chat_template.*`](chat_template.hpp) | BPE vocabulary/merge handling and bounded deterministic Qwen ChatML formatting. |
| [`oracles.*`](oracles.hpp) | Independent reference helpers used for correctness comparison. |
| [`CMakeLists.txt`](CMakeLists.txt) | Explicit Qwen production source registration and HIP source ownership. |

## State and arithmetic

Execution mode is explicit per session. Each request owns positions, KV and
recurrent/convolution state, RNG, proposals and acceptance/controller feedback.
Batch projection work may be shared; state and reduction order must remain
independent. Snapshots bind complete model/projector identities and sampling mode.

Use the model quality runners before changing arithmetic, layouts or cache
boundaries. Production routes compile for gfx1151 through Nix. GPU correctness
builds are optimized with assertions; release binaries provide speed results.
Pinned upstream source/license records remain beside imported kernels and under
`reference/`. Do not add alternate runtime routes without measured benefit.
