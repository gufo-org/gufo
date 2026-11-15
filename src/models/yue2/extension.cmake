# YuE2 music generation extension.

gufo_extension_target(
  NAME gufo_yue2
  SOURCES
    config.cpp
    noise.cpp
    protocol.cpp
    safetensors.cpp
    sampler.cpp
    tokenizer.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
    ICU::uc
)

if(TARGET gufo_yue2)
  target_include_directories(gufo_yue2 PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

gufo_extension_target(
  NAME gufo_yue2_hip
  SOURCES
    hip/backbone.cpp
    hip/vae.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_yue2
    gufo_qwen_kernels
    gufo_qwen_eager
    hip::host
    roc::hipblas
    roc::hipblaslt
)

if(TARGET gufo_yue2_hip)
  gufo_extension_hip_sources(gufo_yue2_hip
    SOURCES
      hip/kernels.hip
  )
  target_compile_definitions(gufo_yue2_hip PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_yue2_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()
