add_library(gufo_moss_image STATIC model.cpp)
target_include_directories(gufo_moss_image PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_moss_image PUBLIC gufo_qwen_image_21 gufo_core)

if(ENGINE_ENABLE_HIP)
  target_sources(gufo_moss_image PRIVATE hip/kernels.hip hip/pipeline.cpp)
  target_link_libraries(gufo_moss_image PUBLIC hip::host)
  # Preserve the BF16 operation boundaries of the PyTorch reference.
  target_compile_options(gufo_moss_image PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffp-contract=off>)
else()
  target_sources(gufo_moss_image PRIVATE model_stub.cpp)
endif()

function(gufo_moss_image_register_tests)
  add_executable(moss_image_test
    ${PROJECT_SOURCE_DIR}/tests/models/moss_image/model_test.cpp)
  target_link_libraries(moss_image_test PRIVATE gufo_moss_image gufo_http)
  add_test(NAME moss_image_test COMMAND moss_image_test)
  set_tests_properties(moss_image_test PROPERTIES LABELS "cpu;models;moss-image")
  if(ENGINE_ENABLE_HIP)
    add_executable(moss_image_kernel_test
      ${PROJECT_SOURCE_DIR}/tests/models/moss_image/kernel_test.hip)
    target_link_libraries(moss_image_kernel_test PRIVATE gufo_moss_image)
    target_compile_options(moss_image_kernel_test PRIVATE
      $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
    add_test(NAME moss_image_kernel_test COMMAND moss_image_kernel_test)
    set_tests_properties(moss_image_kernel_test PROPERTIES
      LABELS "gpu;moss-image" SKIP_RETURN_CODE 77)
    add_executable(moss_image_probe
      ${PROJECT_SOURCE_DIR}/tools/models/moss_image/probe.cpp)
    target_link_libraries(moss_image_probe PRIVATE gufo_moss_image)
  endif()
endfunction()
