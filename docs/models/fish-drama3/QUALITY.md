# Fish Audio Drama-3 quality

**Native HIP operators match FP32 CPU reference implementations; streamed
4-frame causal decoding and offline full-sequence decoding produce bitwise
identical 44.1 kHz waveforms.**

| Check | Result |
| --- | --- |
| Layer 0 native `F8_E4M3` MoE + `BF16` shared expert vs FP32 reference | Cosine 1.0; maximum absolute diff 1.83e-4 |
| Layer 0 register-resident Gated DeltaNet recurrent (4 steps) vs FP32 reference | Cosine 1.0; maximum absolute diff 1.79e-7 |
| Fish DAC 4-frame causal streaming decode vs offline full-sequence decode | Maximum absolute diff **0.0** (bitwise identical) |
| Direct C++ `codec.pth` weight-norm folding vs PyTorch `remove_weight_norm` | Bitwise identical `semantic[0] = 2200` on 16-frame encode |
| Kernel register spill check (`check-kernel-resources.py`) | **0 bytes scratch / 0 spilled VGPRs** across all `fish_drama3` kernels |

## Reproduce

```sh
nix develop -c cmake --build --preset gpu-test --target fish_drama3_config_test fish_drama3_test
nix develop -c ./build/gpu-test/fish_drama3_config_test
nix develop -c ./build/gpu-test/fish_drama3_test
```
