# Pre-migration serialization check needs neither HIP nor model weights:
# it compiles continuation_identity.cpp itself for its own fixture path and
# a HIP-free compile, instead of linking gufo_serving.
add_executable(qwen27b_continuation_identity_test continuation_identity_test.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/continuation_identity.cpp)
target_include_directories(qwen27b_continuation_identity_test PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_definitions(qwen27b_continuation_identity_test PRIVATE
  GUFO_IDENTITY_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/continuation_identity")
add_test(NAME qwen27b_continuation_identity_test COMMAND qwen27b_continuation_identity_test)
set_tests_properties(qwen27b_continuation_identity_test PROPERTIES LABELS "cpu;qwen27b;cache;correctness")

# Model-specific checks live here; generic Qwen operator controls stay in qwen/.
add_test(NAME qwen27b.tools
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools_test.py)
set_tests_properties(qwen27b.tools PROPERTIES LABELS "cpu;qwen27b;tools")

function(add_qwen27b_test target source labels)
  add_executable(${target} ${CMAKE_CURRENT_SOURCE_DIR}/${source})
  target_link_libraries(${target} PRIVATE gufo_core ${GUFO_QWEN_LIBRARIES})
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES LABELS "qwen27b;${labels}"
    SKIP_RETURN_CODE 77)
endfunction()

add_qwen27b_test(qwen27b_vision_test vision_test.cpp "cpu;vision;correctness")
add_qwen27b_test(qwen27b_continuation_layout_test continuation_layout_test.cpp
  "cpu;correctness")

add_qwen27b_test(qwen_mtp_reference_test mtp_reference_test.cpp
  "cpu;mtp;external-model")
if (ENGINE_ENABLE_HIP)
  add_executable(qwen27b_continuation_adapter_test EXCLUDE_FROM_ALL
    continuation_adapter_test.cpp continuation_fault.cpp)
  target_link_libraries(qwen27b_continuation_adapter_test PRIVATE
    gufo_core ${GUFO_QWEN_LIBRARIES} ${CMAKE_DL_LIBS})
  add_qwen27b_test(qwen_sampling_hip_test sampling_test.hip
    "hip;gpu;sampling;correctness")
  target_link_libraries(qwen_sampling_hip_test PRIVATE hip::host)
  # The CLI tests drive the gufo binary, which Windows does not build yet.
  if (TARGET gufo)
    add_test(NAME qwen27b.cli-options COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/cli_test.py $<TARGET_FILE:gufo> --scope options)
    set_tests_properties(qwen27b.cli-options PROPERTIES LABELS "qwen27b;cpu;cli")
    add_test(NAME qwen27b.cli COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/cli_test.py $<TARGET_FILE:gufo>)
    set_tests_properties(qwen27b.cli PROPERTIES
      LABELS "qwen27b;gpu;hip;cli;external-model" SKIP_RETURN_CODE 77)
  endif()
  add_qwen27b_test(qwen27b_target_test target_test.cpp
    "gpu;hip;target;external-model")
  # target_test.cpp reports its logits through the shared comparison support.
  target_link_libraries(qwen27b_target_test PRIVATE gufo_compare)
  add_qwen27b_test(qwen_mtp_gpu_test mtp_test.cpp "gpu;hip;mtp;external-model")
  add_qwen27b_test(qwen_dflash_gpu_test dflash_test.cpp
    "gpu;hip;dflash;dflash2;external-model")
  add_qwen27b_test(qwen_dflash_upload_lifetime_test
    dflash_upload_lifetime_test.cpp "gpu;hip;dflash2;external-model")
  target_link_libraries(qwen_dflash_upload_lifetime_test PRIVATE hip::host)
  target_link_options(qwen_dflash_upload_lifetime_test
    PRIVATE -Wl,--wrap=hipMemcpyAsync)
  add_qwen27b_test(qwen_dflash_noncausal_attention_ops_test attention_test.cpp
    "gpu;hip;dflash;dflash2;kernels")
  target_link_libraries(qwen_dflash_noncausal_attention_ops_test PRIVATE hip::host)
  # The serving tier is one library now (RFC step 6); these tests keep naming
  # the models they drive directly.
  add_executable(inference_backend_gpu_test serving_test.cpp)
  target_link_libraries(inference_backend_gpu_test PRIVATE gufo_serving)
  add_executable(qwen_vision_serving_test ../qwen/vision_serving_test.cpp)
  target_link_libraries(qwen_vision_serving_test PRIVATE gufo_serving)
  add_test(NAME qwen_vision_serving_test COMMAND qwen_vision_serving_test)
  set_tests_properties(qwen_vision_serving_test PROPERTIES
    LABELS "qwen27b;qwen38-flash-next;hip;gpu;vision;external-model"
    SKIP_RETURN_CODE 77)
  add_qwen27b_test(qwen_vision_prepare_test ../qwen/vision_prepare_test.cpp
    "qwen38-flash-next;gpu;hip;vision;external-model")
  target_link_libraries(qwen_vision_prepare_test PRIVATE hip::host)
  target_link_options(qwen_vision_prepare_test PRIVATE -Wl,--wrap=hipMemcpyAsync)
  add_test(NAME inference_backend_gpu_test COMMAND inference_backend_gpu_test)
  set_tests_properties(inference_backend_gpu_test PROPERTIES
    LABELS "qwen27b;hip;gpu;server;correctness;external-model"
    SKIP_RETURN_CODE 77)

endif()
