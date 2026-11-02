# MiniMax H3 FL2VA model extension.

gufo_extension_target(
  NAME gufo_minimax_h3
  SOURCES
    audio_vae.cpp
    denoiser.cpp
    dit.cpp
    generation.cpp
    json.cpp
    media.cpp
    prompt_encoder.cpp
    runtime.cpp
    sampling.cpp
    sha256.cpp
    tokenizer.cpp
    video_vae.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
)

if(TARGET gufo_minimax_h3)
  target_include_directories(gufo_minimax_h3 PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

gufo_extension_target(
  NAME gufo_minimax_h3_hip
  SOURCES
    runtime_hip.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_minimax_h3
    hip::host
    roc::hipblaslt
    roc::rocblas
)

if(TARGET gufo_minimax_h3_hip)
  set_property(SOURCE dit.hip APPEND PROPERTY COMPILE_OPTIONS -O2)
  gufo_extension_hip_sources(gufo_minimax_h3_hip
    SOURCES
      audio_vae.hip
      denoiser.hip
      dit.hip
      dense.hip
      prompt_encoder.hip
      sampling.hip
      video_vae.hip
      attention.hip
  )
  target_compile_definitions(gufo_minimax_h3_hip PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_minimax_h3_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
  target_include_directories(gufo_minimax_h3_hip SYSTEM PRIVATE
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR})
endif()
