# DeepSeek V4 Flash model extension.

gufo_extension_target(
  NAME gufo_deepseek_v4_flash
  SOURCES
    chat_template.cpp
    runtime/sampling.cpp
    tool_calls.cpp
    probe.cpp
    continuation_identity.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
)

if(TARGET gufo_deepseek_v4_flash)
  target_include_directories(gufo_deepseek_v4_flash PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
  target_include_directories(gufo_deepseek_v4_flash SYSTEM PUBLIC
    ${GUFO_EXTENSION_BUILD_ROOT}/runtime
  )
  target_compile_definitions(gufo_deepseek_v4_flash PRIVATE
    _GNU_SOURCE
    DS4_ROCM_BUILD=1)
  if (ENGINE_ENABLE_HIP)
    target_compile_options(gufo_deepseek_v4_flash PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-O3;-ffast-math;-fno-finite-math-only>)
  endif()
endif()

# Compatibility alias for text tier
add_library(gufo_deepseek_v4_flash_text INTERFACE)
target_link_libraries(gufo_deepseek_v4_flash_text INTERFACE gufo_deepseek_v4_flash)

if (ENGINE_ENABLE_HIP)
  set(DEEPSEEK_V4_FLASH_ROCM_DIR "${GUFO_EXTENSION_BUILD_ROOT}/kernels/rocm")
  set(DEEPSEEK_V4_FLASH_MMQ_DIR "${DEEPSEEK_V4_FLASH_ROCM_DIR}/mmq")

  set(DEEPSEEK_V4_FLASH_MMQ_SOURCES
    ${DEEPSEEK_V4_FLASH_MMQ_DIR}/ds4_ggml_stubs.hip.cpp
    ${DEEPSEEK_V4_FLASH_MMQ_DIR}/ds4_mmq.hip.cpp
    ${DEEPSEEK_V4_FLASH_MMQ_DIR}/ds4_mmq_d2r.hip.cpp
    ${DEEPSEEK_V4_FLASH_MMQ_DIR}/mmid.hip.cpp
    ${DEEPSEEK_V4_FLASH_MMQ_DIR}/mmvq.hip.cpp
    ${DEEPSEEK_V4_FLASH_MMQ_DIR}/quantize.hip.cpp)

  add_library(gufo_deepseek_v4_flash_mmq STATIC
    ${DEEPSEEK_V4_FLASH_MMQ_SOURCES})
  set_source_files_properties(${DEEPSEEK_V4_FLASH_MMQ_SOURCES}
    PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_deepseek_v4_flash_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_deepseek_v4_flash_mmq PRIVATE
    GGML_USE_HIP
    GGML_HIP_NO_VMM
    DS4_ROCM_WMMA_MMQ_Y=64)
  target_compile_options(gufo_deepseek_v4_flash_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_deepseek_v4_flash_mmq SYSTEM PUBLIC
    ${DEEPSEEK_V4_FLASH_MMQ_DIR})
  target_link_libraries(gufo_deepseek_v4_flash_mmq PUBLIC
    hip::host
    roc::hipblas)
endif()

gufo_extension_target(
  NAME gufo_deepseek_v4_flash_hip
  SOURCES
    engine.cpp
    runtime/model_data.cpp
    runtime/rocm_graph.cpp
    runtime/session.cpp
    runtime/tokenizer.cpp
    runner.cpp
    drivers.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_deepseek_v4_flash
    gufo_deepseek_v4_flash_mmq
    gufo_logging
    gufo_compare
    hip::host
    roc::hipblas
    roc::hipblaslt
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

if(TARGET gufo_deepseek_v4_flash_hip)
  set(DEEPSEEK_V4_FLASH_ROCM_DIR "${GUFO_EXTENSION_BUILD_ROOT}/kernels/rocm")
  gufo_extension_hip_sources(gufo_deepseek_v4_flash_hip
    SOURCES
      kernels/rocm/backend.hip.cpp
  )
  target_compile_definitions(gufo_deepseek_v4_flash_hip PRIVATE
    _GNU_SOURCE
    DS4_ROCM_BUILD=1)
  target_compile_options(gufo_deepseek_v4_flash_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-ffast-math;-fno-finite-math-only>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_deepseek_v4_flash_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
  target_include_directories(gufo_deepseek_v4_flash_hip SYSTEM PUBLIC
    ${GUFO_EXTENSION_BUILD_ROOT}/runtime
    ${DEEPSEEK_V4_FLASH_ROCM_DIR}
    ${DEEPSEEK_V4_FLASH_ROCM_DIR}/detail
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
endif()
