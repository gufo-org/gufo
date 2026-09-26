# Qwen3.5/3.6-35B-A3B (qwen35moe) target checks. Model-dependent; skips with
# code 77 when GUFO_QWEN35BA3B_MODEL is unset. The op test is synthetic and
# needs no model file.
if (ENGINE_ENABLE_HIP)
  add_executable(qwen35ba3b_moe_ops_test
    ${CMAKE_CURRENT_SOURCE_DIR}/moe_ops_test.cpp)
  target_link_libraries(qwen35ba3b_moe_ops_test
    PRIVATE gufo_core gufo_qwen38_flash_next_mmq)
  add_test(NAME qwen35ba3b_moe_ops_test COMMAND qwen35ba3b_moe_ops_test)
  set_tests_properties(qwen35ba3b_moe_ops_test PROPERTIES
    LABELS "qwen35ba3b;gpu;hip;moe;kernels;correctness")

  add_executable(qwen35ba3b_target_test
    ${CMAKE_CURRENT_SOURCE_DIR}/target_test.cpp)
  target_link_libraries(qwen35ba3b_target_test PRIVATE gufo_core)
  add_test(NAME qwen35ba3b_target_test COMMAND qwen35ba3b_target_test)
  set_tests_properties(qwen35ba3b_target_test PROPERTIES
    LABELS "qwen35ba3b;gpu;hip;target;external-model" SKIP_RETURN_CODE 77)
endif()
