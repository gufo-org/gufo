# Qwen3-TTS 12Hz 1.7B extension.

gufo_extension_target(
  NAME gufo_qwen3_tts
  SOURCES
    audio.cpp
    config.cpp
    loader.cpp
    prompt.cpp
    speaker_features.cpp
    sampling.cpp
    tensor.cpp
    tokenizer.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
    gufo_qwen_control_tokens
    ICU::uc
    Threads::Threads
)

if(TARGET gufo_qwen3_tts)
  target_include_directories(gufo_qwen3_tts PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

gufo_extension_target(
  NAME gufo_qwen3_tts_hip
  SOURCES
    hip/encoder_support.cpp
    hip/speaker_encoder_runtime.cpp
    hip/speech_encoder_runtime.cpp
    hip/speech_decoder_runtime.cpp
    hip/synthesis_runtime.cpp
    hip/talker_runtime.cpp
    npu/speaker_encoder_npu.cpp
    npu/speech_decoder_npu.cpp
    npu/speech_encoder_npu.cpp
    npu/talker_npu.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_qwen3_tts
    gufo_qwen_kernels
    hip::host
    roc::hipblas
    roc::rocblas
)

if(TARGET gufo_qwen3_tts_hip)
  gufo_extension_hip_sources(gufo_qwen3_tts_hip
    SOURCES
      hip/encoder_ops.hip
      hip/speech_decoder_ops.hip
      hip/talker_ops.hip
  )
  target_compile_definitions(gufo_qwen3_tts_hip PUBLIC ENGINE_ENABLE_HIP=1)
  # The NPU runtimes' CPU operators: AVX2 + FMA, as src/core/npu/host_ops.
  set_source_files_properties(
    ${CMAKE_CURRENT_LIST_DIR}/npu/speaker_encoder_npu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/npu/speech_decoder_npu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/npu/speech_encoder_npu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/npu/talker_npu.cpp
    TARGET_DIRECTORY gufo_qwen3_tts_hip
    PROPERTIES COMPILE_OPTIONS
      "$<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX2,-mavx2;-mfma;-O3>")
  target_include_directories(gufo_qwen3_tts_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
    ${GUFO_EXTENSION_BUILD_ROOT}/hip
  )
endif()
