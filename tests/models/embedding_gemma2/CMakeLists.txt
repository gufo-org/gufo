# EmbeddingGemma 2 tests. Only the HIP and external-model checks need weights.

add_executable(embedding_gemma2_config_test config_test.cpp)
target_link_libraries(embedding_gemma2_config_test PRIVATE gufo_embedding_gemma2)
add_test(NAME embedding_gemma2_config_test
  COMMAND embedding_gemma2_config_test)
set_tests_properties(embedding_gemma2_config_test PROPERTIES
  LABELS "cpu;models;embedding-gemma2;config;quality")

add_executable(embedding_gemma2_loader_test loader_test.cpp)
target_link_libraries(embedding_gemma2_loader_test PRIVATE gufo_embedding_gemma2)
add_test(NAME embedding_gemma2_loader_test
  COMMAND embedding_gemma2_loader_test)
set_tests_properties(embedding_gemma2_loader_test PROPERTIES
  LABELS "cpu;models;embedding-gemma2;loader;safetensors;external-model"
  SKIP_RETURN_CODE 77)

add_executable(embedding_gemma2_tokenizer_test tokenizer_test.cpp)
target_link_libraries(embedding_gemma2_tokenizer_test PRIVATE
  gufo_embedding_gemma2)
target_compile_definitions(embedding_gemma2_tokenizer_test PRIVATE
  GUFO_EMBEDDING_GEMMA2_TOKENIZER_GOLDEN="${PROJECT_SOURCE_DIR}/tests/fixtures/embedding_gemma2/tokenizer_golden.json")
add_test(NAME embedding_gemma2_tokenizer_test
  COMMAND embedding_gemma2_tokenizer_test)
set_tests_properties(embedding_gemma2_tokenizer_test PROPERTIES
  LABELS "cpu;models;embedding-gemma2;tokenizer;quality;external-model"
  SKIP_RETURN_CODE 77)
