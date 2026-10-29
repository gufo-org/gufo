# Small host checks are the default. The full-model reference probe is built
# explicitly so routine iterations stay focused.
add_custom_target(nemotron35_bolt_tests)

add_executable(nemotron35_bolt_config_test config_test.cpp)
target_link_libraries(nemotron35_bolt_config_test PRIVATE gufo_nemotron35_bolt)
add_test(NAME nemotron35_bolt.config COMMAND nemotron35_bolt_config_test)
set_tests_properties(nemotron35_bolt.config PROPERTIES
  LABELS "nemotron35_bolt;cpu")
add_dependencies(nemotron35_bolt_tests nemotron35_bolt_config_test)

add_executable(nemotron35_bolt_tokenizer_test tokenizer_test.cpp)
target_link_libraries(nemotron35_bolt_tokenizer_test PRIVATE gufo_nemotron35_bolt)
target_compile_definitions(nemotron35_bolt_tokenizer_test PRIVATE
  GUFO_NEMOTRON35_BOLT_TOKENIZER_GOLDEN="${PROJECT_SOURCE_DIR}/tests/fixtures/nemotron35_bolt/tokenizer_golden.json")
add_test(NAME nemotron35_bolt.pretokenizer
  COMMAND nemotron35_bolt_tokenizer_test pretokenizer)
set_tests_properties(nemotron35_bolt.pretokenizer PROPERTIES
  LABELS "nemotron35_bolt;cpu;tokenizer")
add_test(NAME nemotron35_bolt.tokenizer
  COMMAND nemotron35_bolt_tokenizer_test encode)
set_tests_properties(nemotron35_bolt.tokenizer PROPERTIES
  LABELS "nemotron35_bolt;cpu;tokenizer;external-model"
  SKIP_RETURN_CODE 77)
add_dependencies(nemotron35_bolt_tests nemotron35_bolt_tokenizer_test)

add_executable(nemotron35_bolt_chat_template_test chat_template_test.cpp)
target_link_libraries(nemotron35_bolt_chat_template_test PRIVATE
  gufo_nemotron35_bolt)
target_compile_definitions(nemotron35_bolt_chat_template_test PRIVATE
  GUFO_NEMOTRON35_BOLT_CHAT_GOLDEN="${PROJECT_SOURCE_DIR}/tests/fixtures/nemotron35_bolt/chat_template_golden.json")
add_test(NAME nemotron35_bolt.chat_template
  COMMAND nemotron35_bolt_chat_template_test render)
set_tests_properties(nemotron35_bolt.chat_template PROPERTIES
  LABELS "nemotron35_bolt;cpu;chat_template")
add_test(NAME nemotron35_bolt.chat_template_tokens
  COMMAND nemotron35_bolt_chat_template_test encode)
set_tests_properties(nemotron35_bolt.chat_template_tokens PROPERTIES
  LABELS "nemotron35_bolt;cpu;chat_template;external-model"
  SKIP_RETURN_CODE 77)
add_dependencies(nemotron35_bolt_tests nemotron35_bolt_chat_template_test)

add_executable(nemotron35_bolt_reference_probe EXCLUDE_FROM_ALL
  reference_probe.cpp)
target_link_libraries(nemotron35_bolt_reference_probe PRIVATE
  gufo_nemotron35_bolt_reference)
set_source_files_properties(reference_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  include(${CMAKE_CURRENT_SOURCE_DIR}/mmq.cmake)

  add_executable(nemotron35_bolt_kernel_ops_test kernel_ops_test.cpp)
  target_link_libraries(nemotron35_bolt_kernel_ops_test PRIVATE
    gufo_nemotron35_bolt_kernels gufo_core)
  add_test(NAME nemotron35_bolt.kernel_ops
    COMMAND nemotron35_bolt_kernel_ops_test)
  set_tests_properties(nemotron35_bolt.kernel_ops PROPERTIES
    LABELS "nemotron35_bolt;hip;gpu;gfx1151")
  add_dependencies(nemotron35_bolt_tests nemotron35_bolt_kernel_ops_test)

  add_executable(nemotron35_bolt_gemv_ops_test gemv_ops_test.cpp)
  target_link_libraries(nemotron35_bolt_gemv_ops_test PRIVATE
    gufo_nemotron35_bolt_kernels gufo_core)
  add_test(NAME nemotron35_bolt.gemv_ops COMMAND nemotron35_bolt_gemv_ops_test)
  set_tests_properties(nemotron35_bolt.gemv_ops PROPERTIES
    LABELS "nemotron35_bolt;hip;gpu;gfx1151")
  add_dependencies(nemotron35_bolt_tests nemotron35_bolt_gemv_ops_test)

  add_executable(nemotron35_bolt_drafter_ops_test drafter_ops_test.cpp)
  target_link_libraries(nemotron35_bolt_drafter_ops_test PRIVATE
    gufo_nemotron35_bolt_kernels)
  add_test(NAME nemotron35_bolt.drafter_ops
    COMMAND nemotron35_bolt_drafter_ops_test)
  set_tests_properties(nemotron35_bolt.drafter_ops PROPERTIES
    LABELS "nemotron35_bolt;hip;gpu;gfx1151")
  add_dependencies(nemotron35_bolt_tests nemotron35_bolt_drafter_ops_test)

  add_executable(nemotron35_bolt_runtime_test runtime_test.cpp)
  target_link_libraries(nemotron35_bolt_runtime_test PRIVATE
    gufo_nemotron35_bolt)
  add_test(NAME nemotron35_bolt.runtime COMMAND nemotron35_bolt_runtime_test)
  set_tests_properties(nemotron35_bolt.runtime PROPERTIES
    LABELS "nemotron35_bolt;hip;gpu;gfx1151;external-model;slow"
    SKIP_RETURN_CODE 77 TIMEOUT 600)
  add_dependencies(nemotron35_bolt_tests nemotron35_bolt_runtime_test)

  add_executable(nemotron35_bolt_runtime_probe EXCLUDE_FROM_ALL
    runtime_probe.cpp)
  target_link_libraries(nemotron35_bolt_runtime_probe PRIVATE
    gufo_nemotron35_bolt)

  add_executable(nemotron35_bolt_drafter_probe EXCLUDE_FROM_ALL
    drafter_probe.cpp)
  target_link_libraries(nemotron35_bolt_drafter_probe PRIVATE
    gufo_nemotron35_bolt)
endif()
