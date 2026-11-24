# Fused Qwen3 prefill on the NPU

`include/gufo/ext/experimental/npu.hpp` now also includes
`gufo::npu::Qwen3Prefill` (`src/core/npu/qwen3_prefill.hpp`): a whole Qwen3
dense prefill (28 layers x 2048, 16 query / 8 KV heads of 128, SwiGLU 6144)
in one NPU dispatch. The IRON design in `src/core/npu/kernels/qwen3pre`
(ported from LFM3's prefill) runs every GEMM, residual + RMSNorm, q/k norm +
RoPE and causal attention on the NPU cores, in programs of 256, 512 and 1024
rows (`qwen3pre_<M>.xclbin`, its instruction stream `.bin` and plan
`_plan.txt`).
- `Open(device, dir)` loads every program; `SetLayer` packs a layer's bf16
  weights (norm weights folded); `SetRopeTheta` sets the rotary base.
- `Run(x, n)` prefills n rows from position 0; `Kv(layer, n, k, v)` then
  returns the layer's cache entries (K after its norm and rotation) for
  `Qwen3Decoder::WriteKv`.

`gufo::npu::FindDesign` also finds designs whose programs are named
`<name>_<variant>.xclbin`. `gufo_npu_design` (`cmake/GufoNpuDesigns.cmake`)
takes `OUTPUTS` for designs whose files are not `<program>.xclbin` and
`<program>_dispatch.so`.

Compatibility: additive.
