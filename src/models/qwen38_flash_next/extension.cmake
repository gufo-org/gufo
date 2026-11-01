# Qwen3.8-Flash-Next model extension.

gufo_extension_target(
  NAME gufo_qwen38_flash_next
  SOURCES
    config.cpp
    continuation_layout.cpp
    ngram.cpp
    rope_scaling.cpp
    weights.cpp
    bench_yarn.cpp
    probe.cpp
    continuation_identity.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
    Threads::Threads
)

if(TARGET gufo_qwen38_flash_next)
  target_include_directories(gufo_qwen38_flash_next PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

add_library(gufo_qwen38_flash_next_reference STATIC EXCLUDE_FROM_ALL
  ${GUFO_EXTENSION_BUILD_ROOT}/cpu_ops.cpp
  ${GUFO_EXTENSION_BUILD_ROOT}/reference.cpp)
target_include_directories(gufo_qwen38_flash_next_reference PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${GUFO_EXTENSION_BUILD_ROOT}
)
target_link_libraries(gufo_qwen38_flash_next_reference PUBLIC
  gufo_qwen38_flash_next)
set_source_files_properties(
  ${GUFO_EXTENSION_BUILD_ROOT}/cpu_ops.cpp
  ${GUFO_EXTENSION_BUILD_ROOT}/reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  set(QFN_ROCM_DIR "${GUFO_EXTENSION_BUILD_ROOT}/kernels/rocm")
  set(QFN_MMQ_DIR "${QFN_ROCM_DIR}/mmq")

  set(QFN_MMQ_SOURCES
    ${QFN_MMQ_DIR}/qfn_ggml_stubs.hip.cpp
    ${QFN_MMQ_DIR}/qfn_mmq.hip.cpp
    ${QFN_MMQ_DIR}/mmid.hip.cpp
    ${QFN_MMQ_DIR}/mmvq.hip.cpp
    ${QFN_MMQ_DIR}/moe_wave64.hip.cpp
    ${QFN_MMQ_DIR}/quantize.hip.cpp)

  add_library(gufo_qwen38_flash_next_mmq STATIC ${QFN_MMQ_SOURCES})
  set_source_files_properties(${QFN_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${QFN_MMQ_DIR}/moe_wave64.hip.cpp
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  set_target_properties(gufo_qwen38_flash_next_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_qwen38_flash_next_mmq PRIVATE
    GGML_HIP_NO_VMM ENGINE_ENABLE_HIP=1)
  target_compile_options(gufo_qwen38_flash_next_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen38_flash_next_mmq SYSTEM PUBLIC ${QFN_MMQ_DIR})
  target_link_libraries(gufo_qwen38_flash_next_mmq PUBLIC hip::host roc::hipblas)
endif()

gufo_extension_target(
  NAME gufo_qwen38_flash_next_hip
  SOURCES
    continuation_adapter.cpp
    kernels/rocm/blaslt.cpp
    kernels/rocm/device_model.cpp
    kernels/rocm/executor.cpp
    kernels/rocm/batch.cpp
    engine.cpp
    runner.cpp
    drivers.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_qwen38_flash_next
    gufo_cache_hip
    gufo_qwen38_flash_next_mmq
    gufo_compare
    hip::host
    roc::hipblas
    roc::hipblaslt
)

if(TARGET gufo_qwen38_flash_next_hip)
  set(QFN_ROCM_DIR "${GUFO_EXTENSION_BUILD_ROOT}/kernels/rocm")
  gufo_extension_hip_sources(gufo_qwen38_flash_next_hip
    SOURCES
      kernels/rocm/kernels.hip.cpp
      kernels/rocm/w8a8_wave64.hip
  )
  set_source_files_properties(${QFN_ROCM_DIR}/w8a8_wave64.hip
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  target_compile_definitions(gufo_qwen38_flash_next_hip PRIVATE QFN_ROCM_BUILD=1)
  target_compile_options(gufo_qwen38_flash_next_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen38_flash_next_hip SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
  target_include_directories(gufo_qwen38_flash_next_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()
