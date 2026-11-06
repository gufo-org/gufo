# Qwen3-ASR 1.7B extension.

gufo_extension_target(
  NAME gufo_qwen3_asr
  SOURCES
    audio.cpp
    bpe.cpp
    config.cpp
    loader.cpp
    prompt.cpp
    tensor.cpp
    tokenizer.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
    ICU::uc
    Threads::Threads
)

if(TARGET gufo_qwen3_asr)
  target_include_directories(gufo_qwen3_asr PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

gufo_extension_target(
  NAME gufo_qwen3_asr_hip
  SOURCES
    hip/transcription_runtime.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_qwen3_asr
    hip::host
    roc::hipblas
    roc::hipblaslt
)

if(TARGET gufo_qwen3_asr_hip)
  gufo_extension_hip_sources(gufo_qwen3_asr_hip
    SOURCES
      hip/audio_encoder_runtime.cpp
      hip/audio_ops.hip
      hip/convolution.hip
      hip/blas.cpp
      hip/text_decoder_runtime.cpp
      hip/text_ops.hip
  )
  target_compile_definitions(gufo_qwen3_asr_hip PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_qwen3_asr_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
    ${GUFO_EXTENSION_BUILD_ROOT}/hip
  )
endif()
