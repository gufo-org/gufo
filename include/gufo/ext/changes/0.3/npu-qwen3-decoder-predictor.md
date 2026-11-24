# Qwen3Decoder: the code-predictor stack on the same program

`gufo::npu::Qwen3Decoder` (`src/core/npu/qwen3_decoder.hpp`) also runs a
5-layer stack of hidden 1024 and FFN 3072 (Qwen3-TTS's code predictor) on the
talker's program, without a hardware-context switch:
- `OpenPredictor(talker, dir)`: the predictor stack (`dir`/qwen3dec_p_*) on
  `talker`'s hardware context; its cache holds 16 positions.
- `hidden()`, `layers()`: the stack's hidden size and layer count.
- `SetHead(index, weights, rows, norm)` and `SelectHead(index)`: one of up to 15
  same-sized output heads (the predictor's per-step heads); `SetHead(weights,
  rows, norm)` sets head 0.

The design (`src/core/npu/kernels/qwen3dec`) reads each token's phase count
and dimensions from its runtime parameters, so `qwen3dec.py` writes a second
dispatch library and layout (`qwen3dec_p_dispatch.so`, `qwen3dec_p_layout.txt`)
for the same xclbin.

Compatibility: additive. Existing callers of `Open`, `SetHead` and `Start` are
unchanged.
