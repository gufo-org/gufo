# Family-library tests. They sit with the family they exercise rather than with
# a model, so a change to shared Qwen code has one obvious focused target.

add_executable(qwen_config_test qwen_config_test.cpp)
target_include_directories(qwen_config_test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(qwen_config_test PRIVATE gufo_core gufo_qwen_config)
add_test(NAME qwen_config_test COMMAND qwen_config_test)
set_tests_properties(qwen_config_test PROPERTIES
  LABELS "cpu;families;qwen;config;correctness")
