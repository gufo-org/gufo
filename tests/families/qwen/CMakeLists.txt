# Family-library tests. They sit with the family they exercise rather than with
# a model, so a change to shared Qwen code has one obvious focused target.

add_executable(qwen_config_test qwen_config_test.cpp)
target_include_directories(qwen_config_test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(qwen_config_test PRIVATE gufo_core gufo_qwen_config)
add_test(NAME qwen_config_test COMMAND qwen_config_test)
set_tests_properties(qwen_config_test PROPERTIES
  LABELS "cpu;families;qwen;config;correctness")

add_executable(qwen_draft_vocab_test qwen_draft_vocab_test.cpp)
target_link_libraries(qwen_draft_vocab_test PRIVATE gufo_qwen_draft_vocab)
add_test(NAME qwen_draft_vocab_test COMMAND qwen_draft_vocab_test)
set_tests_properties(qwen_draft_vocab_test PROPERTIES
  LABELS "cpu;families;qwen;speculative;correctness")

add_executable(qwen_vision_admission_test vision_admission_test.cpp)
target_link_libraries(qwen_vision_admission_test PRIVATE gufo_qwen_vision)
add_test(NAME qwen_vision_admission_test COMMAND qwen_vision_admission_test)
set_tests_properties(qwen_vision_admission_test PROPERTIES
  LABELS "cpu;families;qwen;vision;correctness")

if (NOT ENGINE_ENABLE_HIP)
  # Compile the production ownership/positioning code against deterministic
  # allocation/copy faults. The encoder and row kernel alone are substituted.
  add_executable(qwen_vision_lifecycle_test
    vision_lifecycle_test.cpp
    ${PROJECT_SOURCE_DIR}/src/families/qwen/vision/device_input.cpp)
  target_include_directories(qwen_vision_lifecycle_test BEFORE PRIVATE fake_hip)
  target_link_libraries(qwen_vision_lifecycle_test PRIVATE gufo_qwen_vision gufo_logging)
  add_test(NAME qwen_vision_lifecycle_test COMMAND qwen_vision_lifecycle_test)
  set_tests_properties(qwen_vision_lifecycle_test PROPERTIES
    LABELS "cpu;families;qwen;vision;correctness")
endif()
