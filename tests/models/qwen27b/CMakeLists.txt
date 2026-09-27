# Model-specific checks live here; generic Qwen operator controls stay in qwen/.
add_test(NAME qwen27b.tools
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools_test.py)
set_tests_properties(qwen27b.tools PROPERTIES LABELS "cpu;qwen27b;tools")

function(add_qwen27b_test target source labels)
  add_executable(${target} ${CMAKE_CURRENT_SOURCE_DIR}/${source})
  target_link_libraries(${target} PRIVATE gufo_core)
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES LABELS "qwen27b;${labels}"
    SKIP_RETURN_CODE 77)
endfunction()

add_qwen27b_test(qwen27b_vision_test vision_test.cpp "cpu;vision;correctness")

add_qwen27b_test(qwen_mtp_reference_test mtp_reference_test.cpp
  "cpu;mtp;external-model")
if (ENGINE_ENABLE_HIP)
  add_qwen27b_test(qwen_sampling_hip_test sampling_test.hip
    "hip;gpu;sampling;correctness")
  target_link_libraries(qwen_sampling_hip_test PRIVATE hip::host)
  add_test(NAME qwen27b.cli-options COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/cli_test.py $<TARGET_FILE:gufo> --scope options)
  set_tests_properties(qwen27b.cli-options PROPERTIES LABELS "qwen27b;cpu;cli")
  add_test(NAME qwen27b.cli COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/cli_test.py $<TARGET_FILE:gufo>)
  set_tests_properties(qwen27b.cli PROPERTIES
    LABELS "qwen27b;gpu;hip;cli;external-model" SKIP_RETURN_CODE 77)
  add_qwen27b_test(qwen27b_target_test target_test.cpp
    "gpu;hip;target;external-model")
  add_qwen27b_test(qwen_mtp_gpu_test mtp_test.cpp "gpu;hip;mtp;external-model")
  add_qwen27b_test(qwen_dflash_gpu_test dflash_test.cpp
    "gpu;hip;dflash;dflash2;external-model")
  add_qwen27b_test(qwen_dflash_noncausal_attention_ops_test attention_test.cpp
    "gpu;hip;dflash;dflash2;kernels")
  target_link_libraries(qwen_dflash_noncausal_attention_ops_test PRIVATE hip::host)
  add_library(qwen_serving_test_support STATIC
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_disk_store.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_model_runner.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_generation_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/tp_executor.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/inference_backend.cpp)
  target_link_libraries(qwen_serving_test_support PUBLIC gufo_logging
    gufo_core gufo_deepseek_v4_flash gufo_qwen38_flash_next gufo_tp_control
    hip::host)
  add_executable(inference_backend_gpu_test serving_test.cpp)
  target_link_libraries(inference_backend_gpu_test PRIVATE qwen_serving_test_support)
  add_executable(qwen_vision_serving_test ../qwen/vision_serving_test.cpp)
  target_link_libraries(qwen_vision_serving_test PRIVATE qwen_serving_test_support)
  add_test(NAME qwen_vision_serving_test COMMAND qwen_vision_serving_test)
  set_tests_properties(qwen_vision_serving_test PROPERTIES
    LABELS "qwen27b;qwen38-flash-next;hip;gpu;vision;external-model"
    SKIP_RETURN_CODE 77)
  add_test(NAME inference_backend_gpu_test COMMAND inference_backend_gpu_test)
  set_tests_properties(inference_backend_gpu_test PROPERTIES
    LABELS "qwen27b;hip;gpu;server;correctness;external-model"
    SKIP_RETURN_CODE 77)

endif()
