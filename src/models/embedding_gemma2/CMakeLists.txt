# EmbeddingGemma 2 text embeddings: model-private BF16 safetensors runtime.

add_library(gufo_embedding_gemma2 STATIC
  config.cpp
  loader.cpp
  tensor.cpp
  tokenizer.cpp
  tokenizer_loader.cpp
)

target_include_directories(gufo_embedding_gemma2 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_embedding_gemma2 PUBLIC Threads::Threads)
