add_library(gufo_qwen_image_21 STATIC image.cpp weights.cpp tokenizer.cpp)
target_include_directories(gufo_qwen_image_21 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen_image_21 PUBLIC gufo_core ICU::i18n)

if(ENGINE_ENABLE_HIP)
  target_sources(gufo_qwen_image_21 PRIVATE
    hip/kernels.hip hip/attention.hip hip/dense.hip hip/convolution.hip
    hip/pipeline.cpp hip/blas.cpp)
  target_link_libraries(gufo_qwen_image_21 PUBLIC hip::host roc::hipblas roc::hipblaslt)
  # Preserve BF16 operation boundaries and explicit FP32 arithmetic.
  target_compile_options(gufo_qwen_image_21 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffp-contract=off>)
  # Schedule independent FP32 value-product chains without changing arithmetic.
  # Line tables avoid LLVM's full-debug register-pressure crash.
  set_source_files_properties(hip/attention.hip PROPERTIES
    COMPILE_OPTIONS
    "-gline-tables-only;-Xarch_device;-mllvm=-amdgpu-sched-strategy=iterative-ilp")
else()
  target_sources(gufo_qwen_image_21 PRIVATE model_stub.cpp)
endif()

function(gufo_qwen_image_21_register_tests)
  add_executable(qwen_image_21_test
    ${PROJECT_SOURCE_DIR}/tests/models/qwen_image_21/model_test.cpp)
  target_link_libraries(qwen_image_21_test PRIVATE gufo_qwen_image_21 gufo_http)
  add_test(NAME qwen_image_21_test COMMAND qwen_image_21_test)
  set_tests_properties(qwen_image_21_test PROPERTIES LABELS "cpu;models;qwen-image")
  if(ENGINE_ENABLE_HIP)
    add_executable(qwen_image_21_kernel_test
      ${PROJECT_SOURCE_DIR}/tests/models/qwen_image_21/kernel_test.hip)
    target_link_libraries(qwen_image_21_kernel_test PRIVATE gufo_qwen_image_21)
    add_test(NAME qwen_image_21_kernel_test COMMAND qwen_image_21_kernel_test)
    set_tests_properties(qwen_image_21_kernel_test PROPERTIES
      LABELS "gpu;qwen-image" SKIP_RETURN_CODE 77)
    add_executable(qwen_image_21_probe
      ${PROJECT_SOURCE_DIR}/tools/models/qwen_image_21/probe.cpp)
    target_link_libraries(qwen_image_21_probe PRIVATE gufo_qwen_image_21)
  endif()
endfunction()
