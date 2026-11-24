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
#   draft_vocab      header-only speculative draft-head vocabulary rules
#   hip_common       header-only execution/attention policy, RoPE layout,
#                    recurrent state, GEMM routing, quant helpers, hooks
#   config           CPU-only GGUF configuration extraction
#   text             tokenizer + chat template
#   kernels          shared HIP operators (ENGINE_ENABLE_HIP only)
#   vision           vision encoder, image prompt, image cache
#
# gufo::qwen_family bundles them as one experimental extension link target; its
# headers are the forwards under include/gufo/ext/families/qwen/.
add_subdirectory(control_tokens)
add_subdirectory(draft_vocab)
add_subdirectory(hip_common)
add_subdirectory(config)
add_subdirectory(text)
if (ENGINE_ENABLE_HIP)
  add_subdirectory(kernels)
endif()
add_subdirectory(vision)

# The experimental Qwen family extension library. Like gufo_ext_core it adds no
# repository include path: the forwarding headers under
# include/gufo/ext/families/qwen/ and their SURFACE closure come from
# gufo_ext_runtime, which the consumer links, as it does gufo_ext_hip in HIP
# builds. This target links the family implementation.
add_library(gufo_qwen_family INTERFACE)
target_link_libraries(gufo_qwen_family INTERFACE
  $<LINK_ONLY:gufo_qwen_config>
  $<LINK_ONLY:gufo_qwen_text>
  $<LINK_ONLY:gufo_qwen_vision>)
if (ENGINE_ENABLE_HIP)
  target_link_libraries(gufo_qwen_family INTERFACE $<LINK_ONLY:gufo_qwen_kernels>)
endif()
add_library(gufo::qwen_family ALIAS gufo_qwen_family)
