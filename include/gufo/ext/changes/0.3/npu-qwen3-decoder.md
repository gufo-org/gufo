# Fused Qwen3 decode program on the NPU

`include/gufo/ext/experimental/npu.hpp` now also includes `Qwen3Decoder`
(`src/core/npu/qwen3_decoder.hpp`): one decode token of a Qwen3 dense stack
(28 layers, hidden 2048, 16 query and 8 KV heads of 128, SwiGLU 6144,
per-head q/k RMSNorm, NeoX RoPE) per NPU dispatch, on the IRON design in
`src/core/npu/kernels/qwen3dec` (`qwen3dec.xclbin`, `qwen3dec_dispatch.so`,
`qwen3dec_layout.txt`, found with `FindDesign("qwen3dec")`).
- `Open(device, dir)`; `SetLayer(l, Layer)` packs a layer's bf16 weights as
  Q8_0 with its norm weights; `SetHead(weights, rows, norm)` packs an output
  head of up to 152064 rows with the final norm; `SetRopeTheta(theta)`.
- `WriteKv(layer, pos, count, k, k_stride, v, v_stride)` hands prefill's cache
  entries (k after its norm and rotation) to the program's bf16 cache of
  `capacity()` (4096) positions.
- `Start(row, pos)` runs the token whose input row is `row` at `pos`
  asynchronously; `Wait()`, then `logits()` (the head rows) and `Hidden(out)`
  (the final hidden state before the final norm).

`gufo_npu_design` installs every `<name>_*.txt` file a design writes beside
its xclbin.

Compatibility: additive.
