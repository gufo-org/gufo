# HRX integration status

Target: AMD Strix Halo (`gfx1151`)

## Supported route

The supported HRX route is the existing HIP Qwen executor running through
HRX's `libamdhip64.so` compatibility library. The `.#hrx` Nix package provides
that configuration without changing the model executor selected by `bench` or
`serve`.

```bash
nix build .#hrx
./result/bin/strix diagnose
./result/bin/strix bench --model <model.gguf> -p 128 -n 16
```

This route retains the production HIP model implementation, including its
quantized projection formats, attention and Gated DeltaNet semantics, cache
management, sampling, and session state.

## Native HRX/Loom prototype

`src/core/hrx/` contains the native runtime and artifact-loading primitives.
`src/models/qwen/hrx/` currently provides:

- a Qwen3.8-27B shape contract derived from GGUF metadata;
- offset-preserving, bounds-checked HRX buffer bindings for mapped GGUF shards;
- exact typed tensor-payload validation and immutable bindings for the strict
  production Q8_0 route (Q8_0 matrices and embeddings, F32 norms/SSM
  parameters);
- dispatch wrappers for thirteen runtime Loom primitives plus a compile-time
  signed-Q8 decode oracle;
- a native arena with reset and artifact-backed recurrent-state snapshots;
- a fail-closed model/session API and explicit `--qwen-backend hrx-native`
  benchmark selector.

The retained AOT artifact set is:

| Artifact | Implemented scope |
| --- | --- |
| `qwen_fused_swiglu_bf16.fb` | BF16 gate/up projections and SwiGLU activation |
| `qwen_fused_rmsnorm_qkv_bf16.fb` | RMSNorm plus the SSM QKV projection (`10240` rows) |
| `qwen_fused_rope_kv_cache_bf16.fb` | RoPE on separate Q/K values and one-position K/V storage |
| `qwen_fused_down_residual_bf16.fb` | BF16 down projection plus residual |
| `qwen_rmsnorm_f32.fb` | True F32 RMSNorm for one hidden row |
| `qwen_residual_add_f32.fb` | F32 residual add |
| `qwen_swiglu_pointwise_f32.fb` | Pointwise SiLU(gate) × up |
| `qwen_split_q_gate_f32.fb` | Split the 12288-row Q+gate output and apply sigmoid |
| `qwen_copy_f32.fb` | Native device-to-device recurrent-state snapshots |
| `qwen_q8_0_embedding_k5120.fb` | Direct signed Q8_0 token-row decode with f16 block scales |
| `qwen_q8_0_gemv_k5120.fb` | Q8_0 GEMV for hidden-input projections, up to 17408 rows |
| `qwen_q8_0_gemv_k6144.fb` | Q8_0 GEMV for attention/SSM output projections |
| `qwen_q8_0_gemv_k17408.fb` | Q8_0 FFN down projection |

`qwen_q8_0_decode_oracle.fb` is a compile/check artifact with negative signed
bytes and a non-unit f16 scale. `QwenHrxExecutor::PrototypeArtifactsReady()`
means only that the thirteen runtime primitive artifacts loaded. `ModelExecutionReady()` remains false until every
required semantic stage exists; it is the authoritative model-level gate.

## Correct Qwen3.8-27B contract

The native prototype accepts only this fixed shape contract:

- 64 layers in a 3:1 SSM/full-attention pattern:
  - 48 SSM layers;
  - 16 full-attention layers.
- hidden width: `5120`;
- FFN width: `17408`;
- full attention:
  - 24 query heads and 4 KV heads;
  - head dimension `256`, rotary dimension `64`;
  - query width `6144`;
  - Q+gate projection width `12288`;
  - separate K and V widths of `1024` each;
  - diagnostic sum of separate projection rows: `14336`.
- Gated DeltaNet:
  - 16 key heads and 48 recurrent/value heads;
  - key/value dimensions `128/128`;
  - QKV width `10240`;
  - gate width `6144`;
  - alpha and beta widths `48`.
- vocabulary size: `248320`.

The `14336` attention projection row count is diagnostic/allocation metadata.
There is no packed-QKV native artifact: production GGUF full-attention weights
remain separate Q+gate, K, and V tensors.

## Intentionally disabled prototypes

The following earlier sketches were removed from AOT packaging and native
readiness because their names overstated their semantics:

- the DeltaNet sketch used 16 rather than 48 recurrent heads and did not update
  recurrent state according to the Qwen Gated DeltaNet equations;
- the macro attention sketch omitted attention score/value computation and the
  output projection, expected an invalid packed projection, and did not
  implement the Q gate correctly;
- the macro FFN sketch did not implement a complete normalized SwiGLU FFN;
- the final-head sketch did not implement the production final RMSNorm and
  complete logits contract;
- the graph-vs-stream benchmark composed the incomplete macro sketches and was
  therefore not a model-level benchmark.

Failing closed avoids plausible but incorrect native outputs.

## What is required for native end-to-end inference

Native HRX can be requested explicitly with
`--qwen-backend hrx-native`, but the selector deliberately prints the missing
capability list and exits until `ModelExecutionReady()` is true. HIP remains
the default and `serve` does not expose the prototype. Before native execution
can run, it needs all of the following with CPU/HIP parity:

1. target parity for the newly packaged direct Q8_0 embedding and fixed-width
   GEMV artifacts. Model creation rejects every mixed K-quant/BF16 projection
   route with the exact tensor name rather than reinterpreting its bytes;
2. target parity for the staged complete Q8_0 FFN implementation (RMSNorm,
   gate/up GEMVs, SwiGLU, down GEMV, residual, and device copy);
3. separate full-attention Q+gate, K, and V projections;
4. Q/K normalization, RoPE, causal attention, output projection, and residual;
5. the full 48-head Gated DeltaNet flow, including convolution, decay,
   normalized key/query processing, state mutation, output norm/gate, and
   output projection;
6. final RMSNorm, logits, sampling, prompt prefill, token decode, and
   end-to-end target-only parity for fixed prompts and decode traces.

Individual artifact parity must not be reported as native model parity.

## Validation

Run the canonical gates on the supported machine:

```bash
nix build .#hrx
nix build .#checks.x86_64-linux.pr
```

The canonical PR check now also realizes `.#hrx`; `.#checks.x86_64-linux.hrx-compile`
can be used as a focused compile/package gate. For runtime validation, run the
HRX-labelled CTests from the Nix-produced test build with
`--output-on-failure`. Record the exact revision, artifact hashes,
ROCm/XRT/HRX versions, generated artifact list, and numerical error metrics.
Report separately whether a failure occurred during artifact compilation,
loading, typed binding, submission, synchronization, or parity comparison.
