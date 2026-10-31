# Qwen family libraries (RFC step 5): small, acyclic, purpose-named layers that
# more than one Qwen extension needs, so no single family library becomes a
# second core. Every library directory owns a `library.json` manifest whose
# `depends` lists the sibling libraries it includes directly, and
# `tools/ci/check-extension-isolation.py --fail-on-family-graph` compares those
# declarations with the real include graph: an undeclared edge, an unused
# declaration, a cycle, an unassigned header or a family header reaching a model
# directory fails the repository gate.
#
# Layer order (each layer may use the ones above it, never a model):
#   control_tokens   header-only shared token spellings
#   hip_common       header-only execution/attention policy, RoPE layout,
#                    recurrent state, GEMM routing, quant helpers, hooks
#   config           CPU-only GGUF configuration extraction
#   text             tokenizer + chat template
#   kernels          shared HIP operators (ENGINE_ENABLE_HIP only)
#   vision           vision encoder, image prompt, image cache
add_subdirectory(control_tokens)
add_subdirectory(hip_common)
add_subdirectory(config)
add_subdirectory(text)
if (ENGINE_ENABLE_HIP)
  add_subdirectory(kernels)
endif()
add_subdirectory(vision)
