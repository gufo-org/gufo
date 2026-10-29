# Nemotron 3.5 Bolt's model-private quantized matrix kernels, adapted from
# llama.cpp (see VENDOR.md). Included from the model CMakeLists.txt when HIP is
# enabled, so CMAKE_CURRENT_SOURCE_DIR is src/models/nemotron35_bolt.
#
# The tier keeps its own C++17 dialect and GGML feature macros, so it builds
# as a separate target. Each MMQ weight format instantiates in its own
# translation unit to keep the template-heavy build parallel. Consumers include
# src/models/nemotron35_bolt/kernels/rocm/mmq/nb_mmq.h; the vendored headers
# stay private so they cannot shadow another model's ggml.h.
set(NEMOTRON35_BOLT_MMQ_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/mmq")

set(NEMOTRON35_BOLT_MMQ_SOURCES
  ${NEMOTRON35_BOLT_MMQ_DIR}/nb_ggml_stubs.hip.cpp
  ${NEMOTRON35_BOLT_MMQ_DIR}/nb_mmq.hip.cpp
  ${NEMOTRON35_BOLT_MMQ_DIR}/mmq_instance_iq2_xxs.hip.cpp
  ${NEMOTRON35_BOLT_MMQ_DIR}/mmq_instance_nvfp4.hip.cpp
  ${NEMOTRON35_BOLT_MMQ_DIR}/mmq_instance_q8_0.hip.cpp
  ${NEMOTRON35_BOLT_MMQ_DIR}/mmvq.hip.cpp
  ${NEMOTRON35_BOLT_MMQ_DIR}/quantize.hip.cpp)

add_library(gufo_nemotron35_bolt_mmq STATIC ${NEMOTRON35_BOLT_MMQ_SOURCES})

set_source_files_properties(${NEMOTRON35_BOLT_MMQ_SOURCES}
  PROPERTIES LANGUAGE HIP)

set_target_properties(gufo_nemotron35_bolt_mmq PROPERTIES
  HIP_STANDARD 17
  HIP_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(gufo_nemotron35_bolt_mmq PRIVATE
  GGML_USE_HIP
  GGML_HIP_NO_VMM)

target_compile_options(gufo_nemotron35_bolt_mmq PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)

target_include_directories(gufo_nemotron35_bolt_mmq
  PUBLIC ${PROJECT_SOURCE_DIR}
  PRIVATE ${NEMOTRON35_BOLT_MMQ_DIR})

target_link_libraries(gufo_nemotron35_bolt_mmq PUBLIC
  hip::host
  roc::hipblas)
