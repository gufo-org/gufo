# EmbeddingGemma 2 tests. Only the HIP and external-model checks need weights.

if (NOT TARGET gufo_embedding_gemma2)
  return()
endif()

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

if (ENGINE_ENABLE_HIP AND TARGET gufo_embedding_gemma2_hip)
  add_executable(embedding_gemma2_attention_hip_test attention_hip_test.cpp)
  target_link_libraries(embedding_gemma2_attention_hip_test PRIVATE
    gufo_embedding_gemma2_hip hip::host)
  add_test(NAME embedding_gemma2_attention_hip_test
    COMMAND embedding_gemma2_attention_hip_test)
  set_tests_properties(embedding_gemma2_attention_hip_test PROPERTIES
    LABELS "gpu;models;embedding-gemma2;attention;quality"
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE)

  add_executable(embedding_gemma2_encoder_hip_test encoder_hip_test.cpp)
  target_link_libraries(embedding_gemma2_encoder_hip_test PRIVATE
    gufo_embedding_gemma2_hip gufo_embedding_gemma2_reference hip::host)
  add_test(NAME embedding_gemma2_encoder_hip_test
    COMMAND embedding_gemma2_encoder_hip_test)
  set_tests_properties(embedding_gemma2_encoder_hip_test PROPERTIES
    LABELS "gpu;models;embedding-gemma2;encoder;quality;external-model"
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE)

  add_executable(embedding_gemma2_embedding_golden_hip_test
    embedding_golden_hip_test.cpp)
  target_link_libraries(embedding_gemma2_embedding_golden_hip_test PRIVATE
    gufo_embedding_gemma2_hip hip::host)
  target_compile_definitions(embedding_gemma2_embedding_golden_hip_test PRIVATE
    GUFO_EMBEDDING_GEMMA2_EMBEDDING_GOLDEN="${PROJECT_SOURCE_DIR}/tests/fixtures/embedding_gemma2/embedding_golden.json")
  add_test(NAME embedding_gemma2_embedding_golden_hip_test
    COMMAND embedding_gemma2_embedding_golden_hip_test)
  set_tests_properties(embedding_gemma2_embedding_golden_hip_test PROPERTIES
    LABELS "gpu;models;embedding-gemma2;embedding;quality;external-model"
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE)

  add_executable(embedding_gemma2_gemm_bench_hip_test gemm_bench_hip_test.cpp)
  target_link_libraries(embedding_gemma2_gemm_bench_hip_test PRIVATE
    gufo_embedding_gemma2_hip hip::host roc::hipblaslt)
  add_test(NAME embedding_gemma2_gemm_bench_hip_test
    COMMAND embedding_gemma2_gemm_bench_hip_test)
  set_tests_properties(embedding_gemma2_gemm_bench_hip_test PROPERTIES
    LABELS "gpu;models;embedding-gemma2;benchmark;slow"
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE)
endif()

add_executable(embedding_gemma2_batch_scheduler_test batch_scheduler_test.cpp)
target_link_libraries(embedding_gemma2_batch_scheduler_test PRIVATE
  gufo_embedding_gemma2)
add_test(NAME embedding_gemma2_batch_scheduler_test
  COMMAND embedding_gemma2_batch_scheduler_test)
set_tests_properties(embedding_gemma2_batch_scheduler_test PROPERTIES
  LABELS "cpu;models;embedding-gemma2;scheduler"
  TIMEOUT 120)

add_executable(embedding_gemma2_tokenizer_test tokenizer_test.cpp)
target_link_libraries(embedding_gemma2_tokenizer_test PRIVATE
  gufo_embedding_gemma2 gufo_platform)
target_compile_definitions(embedding_gemma2_tokenizer_test PRIVATE
  GUFO_EMBEDDING_GEMMA2_TOKENIZER_GOLDEN="${PROJECT_SOURCE_DIR}/tests/fixtures/embedding_gemma2/tokenizer_golden.json")
add_test(NAME embedding_gemma2_tokenizer_test
  COMMAND embedding_gemma2_tokenizer_test)
set_tests_properties(embedding_gemma2_tokenizer_test PROPERTIES
  LABELS "cpu;models;embedding-gemma2;tokenizer;quality;external-model"
  SKIP_RETURN_CODE 77)

# POST /v1/embeddings against a fake encoder: no weights, no device, no socket.
add_executable(embedding_gemma2_embeddings_api_test
  embeddings_api_test.cpp)
target_link_libraries(embedding_gemma2_embeddings_api_test PRIVATE
  gufo_embedding_gemma2 gufo_test_support gufo_logging gufo_core)
target_compile_definitions(embedding_gemma2_embeddings_api_test PRIVATE
  GUFO_EMBEDDING_GEMMA2_NATIVE_FIXTURE="${PROJECT_SOURCE_DIR}/tests/fixtures/embedding_gemma2/llama_cpp_native_embeddings.json")
add_test(NAME embedding_gemma2_embeddings_api_test
  COMMAND embedding_gemma2_embeddings_api_test)
set_tests_properties(embedding_gemma2_embeddings_api_test PROPERTIES
  LABELS "cpu;models;embedding-gemma2;server;api;quality")

add_executable(embedding_gemma2_image_processor_test image_golden_test.cpp)
target_link_libraries(embedding_gemma2_image_processor_test PRIVATE
  gufo_embedding_gemma2 gufo_core)
target_compile_definitions(embedding_gemma2_image_processor_test PRIVATE
  GUFO_EMBEDDING_GEMMA2_IMAGE_GOLDEN="${PROJECT_SOURCE_DIR}/tests/fixtures/embedding_gemma2/image_golden.json")
add_test(NAME embedding_gemma2_image_processor_test
  COMMAND embedding_gemma2_image_processor_test)
set_tests_properties(embedding_gemma2_image_processor_test PROPERTIES
  LABELS "cpu;models;embedding-gemma2;image;quality")

if (ENGINE_ENABLE_HIP AND TARGET gufo_embedding_gemma2_hip)
  add_executable(embedding_gemma2_image_golden_hip_test image_golden_test.cpp)
  target_link_libraries(embedding_gemma2_image_golden_hip_test PRIVATE
    gufo_embedding_gemma2_hip gufo_core hip::host)
  target_compile_definitions(embedding_gemma2_image_golden_hip_test PRIVATE
    GUFO_IMAGE_TEST_GPU=1
    GUFO_EMBEDDING_GEMMA2_IMAGE_GOLDEN="${PROJECT_SOURCE_DIR}/tests/fixtures/embedding_gemma2/image_golden.json")
  add_test(NAME embedding_gemma2_image_golden_hip_test
    COMMAND embedding_gemma2_image_golden_hip_test)
  set_tests_properties(embedding_gemma2_image_golden_hip_test PROPERTIES
    LABELS "gpu;models;embedding-gemma2;image;quality;external-model"
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE)
endif()

add_executable(embedding_gemma2_audio_processor_test audio_processor_test.cpp)
target_link_libraries(embedding_gemma2_audio_processor_test PRIVATE
  gufo_embedding_gemma2)
target_compile_definitions(embedding_gemma2_audio_processor_test PRIVATE
  GUFO_EMBEDDING_GEMMA2_AUDIO_FEATURES_GOLDEN="${PROJECT_SOURCE_DIR}/tests/fixtures/embedding_gemma2/audio_features_golden.json")
add_test(NAME embedding_gemma2_audio_processor_test
  COMMAND embedding_gemma2_audio_processor_test)
set_tests_properties(embedding_gemma2_audio_processor_test PROPERTIES
  LABELS "cpu;models;embedding-gemma2;audio;quality")

if (ENGINE_ENABLE_HIP AND TARGET gufo_embedding_gemma2_hip)
  add_executable(embedding_gemma2_audio_golden_hip_test audio_golden_hip_test.cpp)
  target_link_libraries(embedding_gemma2_audio_golden_hip_test PRIVATE
    gufo_embedding_gemma2_hip hip::host)
  target_compile_definitions(embedding_gemma2_audio_golden_hip_test PRIVATE
    GUFO_EMBEDDING_GEMMA2_SPEECH_WAV="${PROJECT_SOURCE_DIR}/tests/functional/fixtures/audio/clear-english-voice.wav"
    GUFO_EMBEDDING_GEMMA2_AUDIO_EMBEDDING_GOLDEN="${PROJECT_SOURCE_DIR}/tests/fixtures/embedding_gemma2/audio_embedding_golden.json")
  add_test(NAME embedding_gemma2_audio_golden_hip_test
    COMMAND embedding_gemma2_audio_golden_hip_test)
  set_tests_properties(embedding_gemma2_audio_golden_hip_test PROPERTIES
    LABELS "gpu;models;embedding-gemma2;audio;quality;external-model"
    SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
endif()

add_executable(embedding_gemma2_audio_decoder_test audio_decoder_test.cpp)
target_link_libraries(embedding_gemma2_audio_decoder_test PRIVATE gufo_embedding_gemma2)
target_compile_definitions(embedding_gemma2_audio_decoder_test PRIVATE
  GUFO_EMBEDDING_GEMMA2_AUDIO_DECODER_GOLDEN="${PROJECT_SOURCE_DIR}/tests/fixtures/embedding_gemma2/audio_decoder_golden.json")
add_test(NAME embedding_gemma2_audio_decoder_test COMMAND embedding_gemma2_audio_decoder_test)
set_tests_properties(embedding_gemma2_audio_decoder_test PROPERTIES
  LABELS "cpu;models;embedding-gemma2;audio;quality")

add_executable(embedding_gemma2_video_processor_test video_processor_test.cpp)
target_link_libraries(embedding_gemma2_video_processor_test PRIVATE gufo_embedding_gemma2 gufo_core)
target_compile_definitions(embedding_gemma2_video_processor_test PRIVATE
  GUFO_VIDEO_GOLDEN="${PROJECT_SOURCE_DIR}/tests/fixtures/embedding_gemma2/video_golden.json")
add_test(NAME embedding_gemma2_video_processor_test COMMAND embedding_gemma2_video_processor_test)
set_tests_properties(embedding_gemma2_video_processor_test PROPERTIES LABELS "cpu;models;embedding-gemma2;video;quality")
