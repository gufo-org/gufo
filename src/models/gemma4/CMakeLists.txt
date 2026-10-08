# Gemma 4 text model package (dense and mixture-of-experts gemma4 targets,
# gemma4-assistant MTP drafter). The scalar oracle is built only by explicit
# validation targets.
add_library(gufo_gemma4 STATIC
  chat_template.cpp
  config.cpp
  draft_policy.cpp
  tokenizer.cpp
  tool_syntax.cpp
  vision/prompt.cpp
  vision/weights.cpp
  weights.cpp)

target_include_directories(gufo_gemma4 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_gemma4 PUBLIC gufo_core Threads::Threads)

if (ENGINE_ENABLE_HIP)
  set(GEMMA4_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
  target_sources(gufo_gemma4 PRIVATE
    ${GEMMA4_ROCM_DIR}/kernels.hip.cpp
    ${GEMMA4_ROCM_DIR}/attention_wmma.hip.cpp
    ${GEMMA4_ROCM_DIR}/gemv.hip.cpp
    ${GEMMA4_ROCM_DIR}/wmma_gemv.hip.cpp
    ${GEMMA4_ROCM_DIR}/moe.hip.cpp
    ${GEMMA4_ROCM_DIR}/moe_prefill.hip.cpp
    ${GEMMA4_ROCM_DIR}/device_model.cpp
    ${GEMMA4_ROCM_DIR}/executor.cpp
    vision/encoder.hip
    engine.cpp)
  set_source_files_properties(${GEMMA4_ROCM_DIR}/kernels.hip.cpp
    ${GEMMA4_ROCM_DIR}/attention_wmma.hip.cpp
    ${GEMMA4_ROCM_DIR}/gemv.hip.cpp
    ${GEMMA4_ROCM_DIR}/wmma_gemv.hip.cpp
    ${GEMMA4_ROCM_DIR}/moe.hip.cpp
    ${GEMMA4_ROCM_DIR}/moe_prefill.hip.cpp
    vision/encoder.hip PROPERTIES LANGUAGE HIP)
  # Projections must round identically at every width (verification equals
  # decode), so they are compiled with strict floating-point semantics.
  set_source_files_properties(${GEMMA4_ROCM_DIR}/gemv.hip.cpp
    ${GEMMA4_ROCM_DIR}/wmma_gemv.hip.cpp
    ${GEMMA4_ROCM_DIR}/moe.hip.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
  target_compile_options(gufo_gemma4 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  # Sampled MTP reuses Flash-Next's top-64 proposal kernel and p/q rules.
  target_link_libraries(gufo_gemma4 PUBLIC hip::host roc::hipblas
    gufo_qwen38_flash_next)
endif()

add_library(gufo_gemma4_reference STATIC EXCLUDE_FROM_ALL reference.cpp
  vision/reference.cpp)
target_link_libraries(gufo_gemma4_reference PUBLIC gufo_gemma4)
# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(reference.cpp vision/reference.cpp PROPERTIES
  COMPILE_OPTIONS "-O2")
