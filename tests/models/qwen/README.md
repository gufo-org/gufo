# Qwen tests

The test tree mirrors `src/models/qwen` so a change has an obvious focused
validation target:

- `cpu/` — model-level CPU reference, forward, generator, and SSM behavior.
- `modules/` — narrow contracts for extracted norm, FFN, and SSM modules.
- `hip/` — pure route-policy tests plus gfx1151 kernel and integration tests,
  grouped by attention, FFN, quant, and basic operation ownership.
- `tokenization/` — tokenizer and chat-template behavior.
- `support/` — deterministic Qwen-only fixtures; repository-wide assertions
  and random helpers remain in `tests/testing/test_common.hpp`.

Qwen27B MTP, DFlash2, model replay and optional BF16 reference checks live in
`tests/models/qwen27b` and run through `tools/qwen27b/check.py`.

Compatibility CTest names remain attached to their primary route.
`qwen_gpu_ops_test` now owns dense GEMM/BLAS coverage, while focused targets
avoid compiling or running unrelated kernels:

- basic: `qwen_gpu_ops_test`, `qwen_elementwise_ops_test`;
- attention: `qwen_attention_decode_ops_test`,
  `qwen_attention_long_context_ops_test`, `qwen_attention_fusion_ops_test`,
  `qwen_attention_projection_ops_test`, `qwen_attention_component_ops_test`;
- FFN: `qwen_ffn_fusion_ops_test`, `qwen_ffn_residual_ops_test`;
- quant: `qwen_quant_gemv_ops_test`, `qwen_kquant_gemv_ops_test`,
  `qwen_dequant_ops_test`;
- recurrent/runtime: `qwen_ssm_ops_test`, `qwen_graph_prefetch_ops_test`,
  `qwen_module_ops_test`.

Small utilities under `hip/support/` provide move-only device allocation,
host/device copies, explicit device requirements, BF16 conversion, and
always-on numeric checks. They intentionally do not replace CTest or introduce
a test registry.

The focused basic, attention, FFN, and quant kernel executables declare HIP
hardware optional at their device gate. No visible HIP device therefore returns
CTest skip code 77 rather than success. HIP runtime discovery errors and tests
that declare hardware required return failure. The shared CMake helper records
77 as the skip code for every focused HIP target; targets without an explicit
device gate still run normally and cannot skip merely because the property is
present.

Labels add `qwen` and the relevant tier, allowing focused runs on a supported
Linux x86-64 Strix Halo host:

```sh
nix develop -c ctest --test-dir build/gpu-test -L qwen --output-on-failure
nix develop -c ctest --test-dir build/gpu-test \
  -L qwen -L module --output-on-failure
nix develop -c ctest --test-dir build/gpu-test \
  -L qwen -L hip --output-on-failure
```

`hip/attention_policy_test.cpp`, `hip/execution_policy_test.cpp`, and
`gemm_route_test.cpp` exercise pure route selection and do not require a GPU
even though they mirror HIP composition. The other files under `hip/` require
the supported gfx1151 target.

The module-seam integration test does not yet cover the complete production
executor: embedding, fused attention/SSM composition, unembedding, sampling,
and real executor/logit parity remain separate acceptance work. Production
performance validation uses the release `nix build` binaries. The `gpu-test`
preset uses optimized code with symbols and keeps test assertions enabled.
