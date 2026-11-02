# Qwen3.8-27B model extension.

gufo_extension_target(
  NAME gufo_qwen
  SOURCES
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
    dflash_weights.cpp
    probe.cpp
    continuation_identity.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
    gufo_qwen_config
    gufo_qwen_control_tokens
    gufo_qwen_hip_common
    gufo_qwen_text
)

if(TARGET gufo_qwen)
  target_include_directories(gufo_qwen PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

gufo_extension_target(
  NAME gufo_qwen_hip
  SOURCES
    continuation_adapter.cpp
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
    hip/dflash/backend.cpp
    runner.cpp
    drivers.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_qwen
    gufo_cache_hip
    gufo_qwen_kernels
    gufo_qwen_vision
    gufo_compare
    hip::host
)

if(TARGET gufo_qwen_hip)
  gufo_extension_hip_sources(gufo_qwen_hip
    SOURCES
      hip/attention.hip
      hip/batched_ssm.hip
  )
  target_compile_definitions(gufo_qwen_hip PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_qwen_hip SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR})
  target_include_directories(gufo_qwen_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

if(TARGET gufo_qwen_hip)
  add_library(gufo_qwen27b INTERFACE)
  target_link_libraries(gufo_qwen27b INTERFACE gufo_qwen_hip)
elseif(TARGET gufo_qwen)
  add_library(gufo_qwen27b INTERFACE)
  target_link_libraries(gufo_qwen27b INTERFACE gufo_qwen)
endif()

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
