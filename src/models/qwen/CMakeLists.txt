# Qwen3.8-27B model library: target inference, native MTP and DFlash2.
#
# Everything shared with other Qwen extensions lives in src/families/qwen/ and
# is named explicitly below. This directory stays the model's home until Card 20
# turns it into an extension with its own extension.json manifest, so the
# library here is a plain target rather than a registered extension.
add_library(gufo_qwen27b STATIC
  continuation_layout.cpp
  oracles.cpp
  state.cpp
  weights.cpp
  ssm.cpp
  forward.cpp
  modules/attention.cpp
  modules/embed.cpp
  modules/ffn.cpp
  modules/norm.cpp
  modules/quant_gemm.cpp
  modules/residual.cpp
  modules/rope.cpp
  modules/sample.cpp
  modules/ssm.cpp
  modules/unembed.cpp
  generator.cpp
  mtp_reference.cpp
  dflash_weights.cpp)

target_include_directories(gufo_qwen27b PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen27b PUBLIC
  gufo_core
  gufo_qwen_config
  gufo_qwen_control_tokens
  gufo_qwen_hip_common
  gufo_qwen_text)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_qwen27b PRIVATE
    continuation_adapter.cpp
    hip/attention.hip
    hip/batched_ssm.hip
    hip/arena.cpp
    hip/batched_decode.cpp
    hip/decode.cpp
    hip/decode_step.cpp
    hip/executor.cpp
    hip/model_loader.cpp
    hip/prefill.cpp
    hip/prefill_chunk.cpp
    hip/prompt_hidden_buffer.cpp
    hip/qwen_speculative_verifier.cpp
    hip/ssm_replay.cpp
    hip/mtp/model.cpp
    hip/mtp/executor.cpp
    hip/mtp/backend.cpp
    hip/dflash/model.cpp
    hip/dflash/executor.cpp
    hip/dflash/batch.cpp
    hip/dflash/backend.cpp)
  # gufo_core kept these device headers private, so the model repeats them.
  target_include_directories(gufo_qwen27b SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR})
  target_link_libraries(gufo_qwen27b PUBLIC
    gufo_cache_hip
    gufo_qwen_kernels
    gufo_qwen_vision)
endif()

# Complete Qwen link set for test and tool executables that exercise the family
# and the 27B model together. gufo_qwen27b propagates the family libraries it
# needs, and the family-only entries keep a CPU-only test honest. Production
# libraries and executables name the Qwen libraries they actually use instead.
set(GUFO_QWEN_LIBRARIES
  gufo_qwen27b
  gufo_qwen_config
  gufo_qwen_control_tokens
  gufo_qwen_hip_common
  gufo_qwen_text
  gufo_qwen_vision
  CACHE INTERNAL "Qwen libraries a test or tool links")

if (ENGINE_ENABLE_HIP AND GUFO_BUILD_TOOLS)
  add_executable(qwen27b_continuation_probe EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tests/models/qwen27b/continuation_probe.cpp)
  target_link_libraries(qwen27b_continuation_probe PRIVATE ${GUFO_QWEN_LIBRARIES})
  add_executable(qwen27b_verify_sweep EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/qwen27b/verify_sweep/sweep.hip)
  target_link_libraries(qwen27b_verify_sweep PRIVATE ${GUFO_QWEN_LIBRARIES} hip::device)
endif()
