# Nemotron 3.5 Bolt 74B-A9B (nemotron_h_moe) model package. The scalar oracle
# is built only by explicit validation targets, not by production or routine
# checks.
add_library(gufo_nemotron35_bolt STATIC
  chat_template.cpp
  config.cpp
  tokenizer.cpp
  weights.cpp)

target_include_directories(gufo_nemotron35_bolt PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_nemotron35_bolt PUBLIC gufo_core ICU::uc)

add_library(gufo_nemotron35_bolt_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_nemotron35_bolt_reference PUBLIC
  gufo_nemotron35_bolt)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  # llama.cpp-derived quantized matrix kernels (gufo_nemotron35_bolt_mmq).
  include(${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/mmq/mmq.cmake)

  # Model-private trunk kernels other than the quantized matrix products.
  add_library(gufo_nemotron35_bolt_kernels STATIC
    kernels/rocm/drafter_kernels.hip
    kernels/rocm/gemm.hip
    kernels/rocm/gemv.hip
    kernels/rocm/kernels.hip
    kernels/rocm/sample.hip)
  set_source_files_properties(kernels/rocm/drafter_kernels.hip
    kernels/rocm/gemm.hip kernels/rocm/gemv.hip kernels/rocm/kernels.hip
    kernels/rocm/sample.hip PROPERTIES LANGUAGE HIP)
  target_include_directories(gufo_nemotron35_bolt_kernels PUBLIC
    ${PROJECT_SOURCE_DIR})
  target_compile_options(gufo_nemotron35_bolt_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_nemotron35_bolt_kernels PUBLIC hip::host)

  target_sources(gufo_nemotron35_bolt PRIVATE
    drafter.cpp
    engine.cpp
    speculative.cpp
    kernels/rocm/device_model.cpp)
  target_link_libraries(gufo_nemotron35_bolt PUBLIC
    gufo_nemotron35_bolt_kernels gufo_nemotron35_bolt_mmq hip::host
    roc::hipblas)
endif()
