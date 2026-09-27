# Qwen3.8-Flash-Next (qwen4exp) model package. The scalar oracle is built
# only by explicit validation targets, not by production or routine checks.
option(GUFO_ENABLE_TP2_RDMA
  "Build the Qwen3.8-Flash-Next TP=2 native RDMA adapter" OFF)

add_library(gufo_qwen38_flash_next STATIC
  config.cpp
  ngram.cpp
  weights.cpp
  distributed/tp_partition.cpp)

target_include_directories(gufo_qwen38_flash_next PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen38_flash_next PUBLIC gufo_core Threads::Threads)

if (GUFO_ENABLE_TP2_RDMA)
  if (NOT ENGINE_ENABLE_HIP)
    message(FATAL_ERROR
      "GUFO_ENABLE_TP2_RDMA requires ENGINE_ENABLE_HIP")
  endif()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(IBVERBS REQUIRED IMPORTED_TARGET libibverbs)
  add_library(gufo_qwen38_flash_next_rdma STATIC
    kernels/rocm/verbs.cpp)
  target_compile_features(gufo_qwen38_flash_next_rdma PRIVATE cxx_std_20)
  target_include_directories(gufo_qwen38_flash_next_rdma PRIVATE
    ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_qwen38_flash_next_rdma PRIVATE
    PkgConfig::IBVERBS Threads::Threads ${CMAKE_DL_LIBS} hip::host)
  target_link_libraries(gufo_qwen38_flash_next PRIVATE
    gufo_qwen38_flash_next_rdma)
  target_compile_definitions(gufo_qwen38_flash_next PUBLIC
    GUFO_ENABLE_TP2_RDMA=1)
endif()

add_library(gufo_qwen38_flash_next_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_qwen38_flash_next_reference PUBLIC
  gufo_qwen38_flash_next)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  set(QFN_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
  set(QFN_MMQ_DIR "${QFN_ROCM_DIR}/mmq")

  # HIP quantized matrix kernels adapted from llama.cpp, isolated in the
  # qfn_mmq namespace. The adapter retains its C++17 dialect.
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
    GGML_HIP_NO_VMM)
  target_compile_options(gufo_qwen38_flash_next_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen38_flash_next_mmq SYSTEM PUBLIC ${QFN_MMQ_DIR})
  target_link_libraries(gufo_qwen38_flash_next_mmq PUBLIC hip::host roc::hipblas)

  set(QFN_HIP_SOURCES
    ${QFN_ROCM_DIR}/kernels.hip.cpp
    ${QFN_ROCM_DIR}/w8a8_wave64.hip
    ${QFN_ROCM_DIR}/blaslt.cpp
    ${QFN_ROCM_DIR}/device_model.cpp
    ${QFN_ROCM_DIR}/executor.cpp
    ${QFN_ROCM_DIR}/batch.cpp
    engine.cpp)
  target_sources(gufo_qwen38_flash_next PRIVATE ${QFN_HIP_SOURCES})
  set_source_files_properties(${QFN_ROCM_DIR}/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${QFN_ROCM_DIR}/w8a8_wave64.hip
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  target_compile_definitions(gufo_qwen38_flash_next PRIVATE QFN_ROCM_BUILD=1)
  target_compile_options(gufo_qwen38_flash_next PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen38_flash_next SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
  target_link_libraries(gufo_qwen38_flash_next PUBLIC
    gufo_qwen38_flash_next_mmq hip::host roc::hipblas roc::hipblaslt)
endif()
