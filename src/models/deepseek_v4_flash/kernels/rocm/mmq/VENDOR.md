# HIP quantized matrix kernels

These model-private kernels derive from llama.cpp commit
`5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` (MIT, The ggml authors),
originally under `ggml/src/ggml-cuda/`. The DS4 adapter also carries changes
from `xangel82/DS4-GB10-GX10-DSpark-CUDA`, commit `910501e`.

Gufo builds this directory with HIP for AMD `gfx1151`. Device headers use
`.hip.hpp`; translation units use `.hip.cpp`. The adapter calls HIP/hipBLAS
directly and uses `ds4_ggml_hip_*` symbols to remain separate from Qwen's
matrix backend. There is no CUDA build or CUDA-to-HIP API alias layer.

`ds4_mmq.h` exposes the raw-pointer C ABI. `ds4_ggml_stubs.h` supplies the
small ggml type and allocation interface needed by the templates. Quantization
layouts, tables and active AMD arithmetic remain from the imported kernels.
Unused NVIDIA/MUSA preprocessing paths, ggml graph support and full ggml
tensor entry points are removed; DS4 owns its HIP graph execution separately.
The C ABI keeps only what the DS4 runtime calls: dense Q8_0 and the paired
IQ2_XXS MoE matmuls.

When importing upstream changes, retain the HIP implementation and rerun the
focused DS4 projection/attention checks through Nix. The backend and all
matrix translation units were also compared before and after preprocessing
for `gfx1151`; only private symbol names and unsupported-kernel diagnostics
changed. Current measurements belong in the model benchmark README.
