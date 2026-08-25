# HRX Qwen3.8-27B feature-parity investigation

## Baseline

- Revision: `e7d6cd7` plus working-tree integration changes.
- Target: native HRX route for the fixed Qwen3.8-27B ABI.
- Current state: HRX runtime packaging and standalone Loom artifact tests pass,
  but the artifacts are not ABI-compatible with the production model.

## ABI findings

- Production Qwen3.8-27B: 64 layers, hidden=5120, FFN=17408,
  attention=24x256=6144, KV heads=4, vocabulary=248320.
- Existing native HRX artifacts use incompatible constants including qkv=8192
  and vocabulary=152064. They are not eligible for CLI routing.

## Next implementation slice

1. Establish the model contract and model-to-HRX-buffer binding.
2. Replace incompatible prototype artifact ABIs before attempting dispatch.
3. Add decode parity against the HIP reference before any benchmark claim.

## Validation

- `qwen_hrx_model_contract_test` passes in an HRX-enabled CMake build.
- Existing `hrx_backend_test` and `qwen_hrx_executor_test` also pass after
  building their explicit targets.
- The corrected native `RMSNorm + SSM-QKV` artifact is compiled for the real
  10,240-row Qwen3.8-27B SSM projection and matches its CPU oracle for a
  two-row BF16 dispatch.
- The production Q8_0 model cannot be dispatched by the existing BF16-only
  Loom prototypes. A native production route needs quantized-weight kernels or
  an explicit BF16-only package contract.
