# Metadata, tokenizer and chat-rendering checks are CPU-only and run in the
# routine suite. Operator and model checks live in the rocm_* tests.
add_custom_target(glm5_next_tests)

add_executable(glm5_next_tokenizer_test tokenizer_test.cpp)
target_link_libraries(glm5_next_tokenizer_test PRIVATE gufo_glm5_next)
add_test(NAME glm5_next.tokenizer COMMAND glm5_next_tokenizer_test)
set_tests_properties(glm5_next.tokenizer PROPERTIES LABELS "glm5_next;cpu")
add_dependencies(glm5_next_tests glm5_next_tokenizer_test)

add_executable(glm5_next_chat_template_test chat_template_test.cpp)
target_link_libraries(glm5_next_chat_template_test PRIVATE gufo_glm5_next)
add_test(NAME glm5_next.chat_template COMMAND glm5_next_chat_template_test)
set_tests_properties(glm5_next.chat_template PROPERTIES
  LABELS "glm5_next;cpu")
add_dependencies(glm5_next_tests glm5_next_chat_template_test)

# Operator and model checks run on the GPU (exclusive access) against the
# real shards and the self-regression fixture; see the model docs.
if(ENGINE_ENABLE_HIP)
  add_executable(glm5_next_rocm_gemv_test EXCLUDE_FROM_ALL
    rocm_gemv_test.cpp)
  target_link_libraries(glm5_next_rocm_gemv_test PRIVATE
    gufo_glm5_next_kernels)
  add_test(NAME glm5_next.rocm_gemv COMMAND glm5_next_rocm_gemv_test)
  set_tests_properties(glm5_next.rocm_gemv PROPERTIES
    LABELS "glm5_next;hip;gpu;gfx1151")

  add_executable(glm5_next_routed_wmma_test EXCLUDE_FROM_ALL
    routed_wmma_test.cpp)
  target_link_libraries(glm5_next_routed_wmma_test PRIVATE
    gufo_glm5_next_kernels)
  add_test(NAME glm5_next.rocm_routed_wmma COMMAND glm5_next_routed_wmma_test)
  set_tests_properties(glm5_next.rocm_routed_wmma PROPERTIES
    LABELS "glm5_next;hip;gpu;gfx1151")

  add_executable(glm5_next_dense_wmma_test EXCLUDE_FROM_ALL
    dense_wmma_test.cpp)
  target_link_libraries(glm5_next_dense_wmma_test PRIVATE
    gufo_glm5_next_kernels)
  add_test(NAME glm5_next.rocm_dense_wmma COMMAND glm5_next_dense_wmma_test)
  set_tests_properties(glm5_next.rocm_dense_wmma PROPERTIES
    LABELS "glm5_next;hip;gpu;gfx1151")

  add_executable(glm5_next_rocm_ops_test EXCLUDE_FROM_ALL
    rocm_ops_test.cpp)
  target_link_libraries(glm5_next_rocm_ops_test PRIVATE
    gufo_glm5_next_kernels)
  add_test(NAME glm5_next.rocm_ops COMMAND glm5_next_rocm_ops_test)
  set_tests_properties(glm5_next.rocm_ops PROPERTIES
    LABELS "glm5_next;hip;gpu;gfx1151")

  add_executable(glm5_next_rocm_attention_test EXCLUDE_FROM_ALL
    rocm_attention_test.cpp)
  target_link_libraries(glm5_next_rocm_attention_test PRIVATE
    gufo_glm5_next_kernels)
  add_test(NAME glm5_next.rocm_attention COMMAND
    glm5_next_rocm_attention_test)
  set_tests_properties(glm5_next.rocm_attention PROPERTIES
    LABELS "glm5_next;hip;gpu;gfx1151")

  add_executable(glm5_next_rocm_device_model_test EXCLUDE_FROM_ALL
    rocm_device_model_test.cpp)
  target_link_libraries(glm5_next_rocm_device_model_test PRIVATE
    gufo_glm5_next_executor)
  add_test(NAME glm5_next.rocm_device_model COMMAND
    glm5_next_rocm_device_model_test)
  set_tests_properties(glm5_next.rocm_device_model PROPERTIES
    LABELS "glm5_next;hip;gpu;gfx1151")

  add_executable(glm5_next_rocm_forward_test EXCLUDE_FROM_ALL
    rocm_forward_test.cpp)
  target_link_libraries(glm5_next_rocm_forward_test PRIVATE
    gufo_glm5_next_executor)
  add_test(NAME glm5_next.rocm_forward COMMAND glm5_next_rocm_forward_test)
  set_tests_properties(glm5_next.rocm_forward PROPERTIES
    ENVIRONMENT
    "GUFO_GLM5_NEXT_FIXTURE=${PROJECT_SOURCE_DIR}/tests/fixtures/glm_next/llama_reference.json"
    LABELS "glm5_next;hip;gpu;gfx1151")

  add_dependencies(glm5_next_tests glm5_next_rocm_gemv_test
    glm5_next_routed_wmma_test glm5_next_dense_wmma_test
    glm5_next_rocm_ops_test glm5_next_rocm_attention_test
    glm5_next_rocm_device_model_test glm5_next_rocm_forward_test)
endif()
