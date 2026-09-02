set(DEEPSEEK_V4_FLASH_RUNTIME_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/runtime")
set(DEEPSEEK_V4_FLASH_ROCM_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
set(DEEPSEEK_V4_FLASH_MMQ_DIR
  "${DEEPSEEK_V4_FLASH_ROCM_DIR}/mmq")

# Vendored llama.cpp quantized-matmul tier. It carries its own C++17 dialect,
# GGML feature macros, and tile-size knob, so it is compiled as a separate
# target instead of leaking those into the model backend.
# See kernels/rocm/mmq/VENDOR.md for the upstream pin and file inventory.
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

target_include_directories(gufo_deepseek_v4_flash_mmq
  SYSTEM PUBLIC
    ${DEEPSEEK_V4_FLASH_MMQ_DIR})

target_link_libraries(gufo_deepseek_v4_flash_mmq PUBLIC
  hip::host
  roc::hipblas)

add_library(gufo_deepseek_v4_flash STATIC
  chat_template.cpp
  engine.cpp
  ${DEEPSEEK_V4_FLASH_RUNTIME_DIR}/model_data.cpp
  ${DEEPSEEK_V4_FLASH_RUNTIME_DIR}/rocm_graph.cpp
  ${DEEPSEEK_V4_FLASH_RUNTIME_DIR}/sampling.cpp
  ${DEEPSEEK_V4_FLASH_RUNTIME_DIR}/session.cpp
  ${DEEPSEEK_V4_FLASH_RUNTIME_DIR}/tokenizer.cpp
  ${DEEPSEEK_V4_FLASH_ROCM_DIR}/backend.hip.cpp)

set_source_files_properties(
  ${DEEPSEEK_V4_FLASH_ROCM_DIR}/backend.hip.cpp
  PROPERTIES LANGUAGE HIP)

target_compile_definitions(gufo_deepseek_v4_flash PRIVATE
  _GNU_SOURCE
  DS4_ROCM_BUILD=1)

target_compile_options(gufo_deepseek_v4_flash PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-ffast-math;-fno-finite-math-only>
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)

target_include_directories(gufo_deepseek_v4_flash
  PUBLIC
    ${PROJECT_SOURCE_DIR}
  SYSTEM PUBLIC
    ${DEEPSEEK_V4_FLASH_RUNTIME_DIR}
    ${DEEPSEEK_V4_FLASH_ROCM_DIR}
    ${DEEPSEEK_V4_FLASH_ROCM_DIR}/detail
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})

target_link_libraries(gufo_deepseek_v4_flash PUBLIC
  gufo_deepseek_v4_flash_mmq
  hip::host
  roc::hipblas
  roc::hipblaslt
  Threads::Threads
  ${CMAKE_DL_LIBS})
