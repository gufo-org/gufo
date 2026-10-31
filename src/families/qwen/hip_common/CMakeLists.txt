# The shared Qwen HIP policy/layout layer, below both qwen_kernels and
# qwen_vision. It resolves the cycles the old src/models/qwen header graph had
# between layers: hip/ops/token.hpp and hip/ops/attention.hpp reached the vision
# RoPE layout, hip/kernels/* reached the recurrent-state and attention-policy
# details, and vision/device_input.hpp reached the continuation hooks.
#
# Header-only on purpose: it compiles no object of its own, so every consumer
# compiles these headers with its own per-source options (including the wave-32
# versus wave-64 kernel split) and production code objects stay identical.
add_library(gufo_qwen_hip_common INTERFACE)
target_include_directories(gufo_qwen_hip_common INTERFACE ${PROJECT_SOURCE_DIR})
