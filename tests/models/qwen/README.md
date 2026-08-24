# Qwen tests

The test tree mirrors `src/models/qwen` so a change has an obvious focused
validation target:

- `cpu/` — model-level CPU reference, forward, generator, and SSM behavior.
- `modules/` — narrow contracts for extracted norm, FFN, and SSM modules.
- `hip/` — pure route-policy tests plus gfx1151 kernel and integration tests.
- `mtp/` — multi-token-prediction reference behavior.
- `tokenization/` — tokenizer and chat-template behavior.
- `xdna2/` — MTP kernels executed through the XDNA2 runtime.
- `support/` — deterministic Qwen-only fixtures; repository-wide assertions
  and random helpers remain in `tests/testing/test_common.hpp`.

Existing CTest executable names remain stable. The former monolithic
`qwen_gpu_ops_test` keeps that name for foundational ops/BLAS coverage, while
focused kernel-family executables make GPU iteration cheaper:

- `qwen_attention_decode_ops_test`
- `qwen_attention_fusion_ops_test`
- `qwen_ssm_ops_test`
- `qwen_ffn_fusion_ops_test`
- `qwen_graph_prefetch_ops_test`
- `qwen_quant_gemv_ops_test`
- `qwen_dequant_ops_test`
- `qwen_module_ops_test`

Labels add `qwen` and the relevant tier, allowing focused runs on a supported
Linux x86-64 Strix Halo host:

```sh
ctest --test-dir build/gpu-test -L qwen --output-on-failure
ctest --test-dir build/gpu-test -L qwen -L module --output-on-failure
ctest --test-dir build/gpu-test -L qwen -L hip --output-on-failure
```

`hip/attention_policy_test.cpp` and `hip/execution_policy_test.cpp` exercise
pure route selection and do not require a GPU even though they mirror HIP
composition. The other files under `hip/` require the supported gfx1151 target.

The module-seam integration test does not yet cover the complete production
executor: embedding, fused attention/SSM composition, unembedding, sampling,
and real executor/logit parity remain separate acceptance work. Production
model and performance validation must use the optimized `nix build` binaries,
not the unoptimized `build/gpu-test` preset.
