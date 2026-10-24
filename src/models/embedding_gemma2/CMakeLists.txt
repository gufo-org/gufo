# EmbeddingGemma 2 text and image embeddings: model-private BF16 safetensors runtime.

add_library(gufo_embedding_gemma2 STATIC
  image_processor.cpp
  config.cpp
  loader.cpp
  tensor.cpp
  tokenizer.cpp
  tokenizer_loader.cpp
)

target_include_directories(gufo_embedding_gemma2 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_embedding_gemma2 PUBLIC Threads::Threads)

# Float32 CPU oracle for the HIP runtime; tests link it explicitly.
add_library(gufo_embedding_gemma2_reference STATIC EXCLUDE_FROM_ALL
  reference/cpu_encoder.cpp
)
target_link_libraries(gufo_embedding_gemma2_reference PUBLIC
  gufo_embedding_gemma2)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_embedding_gemma2 PRIVATE
    hip/blas.cpp
    hip/encoder_runtime.cpp
    hip/ops.hip
    hip/vision_ops.hip)
  target_compile_definitions(gufo_embedding_gemma2 PUBLIC ENGINE_ENABLE_HIP=1)
  target_link_libraries(gufo_embedding_gemma2 PUBLIC
    hip::host
    roc::hipblas
    roc::hipblaslt)
endif()
