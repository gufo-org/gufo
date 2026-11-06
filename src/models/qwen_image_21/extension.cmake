# Qwen-Image-2.1 extension.

gufo_extension_target(
  NAME gufo_qwen_image_21
  SOURCES
    image.cpp
    weights.cpp
    tokenizer.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
    gufo_qwen_text
    ICU::i18n
)

if(TARGET gufo_qwen_image_21)
  target_include_directories(gufo_qwen_image_21 PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

gufo_extension_target(
  NAME gufo_qwen_image_21_hip
  SOURCES
    provider.cpp
  LINK_LIBRARIES
    gufo_qwen_image_21
    hip::host
    roc::hipblas
    roc::hipblaslt
)

if(TARGET gufo_qwen_image_21_hip)
  gufo_extension_hip_sources(gufo_qwen_image_21_hip
    SOURCES
      hip/kernels.hip
      hip/attention.hip
      hip/dense.hip
      hip/convolution.hip
      hip/pipeline.cpp
      hip/blas.cpp
  )
  target_compile_definitions(gufo_qwen_image_21_hip PUBLIC ENGINE_ENABLE_HIP=1)
  target_compile_options(gufo_qwen_image_21_hip PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffp-contract=off>)
  set_source_files_properties(${GUFO_EXTENSION_BUILD_ROOT}/hip/attention.hip PROPERTIES
    COMPILE_OPTIONS
    "-gline-tables-only;-Xarch_device;-mllvm=-amdgpu-sched-strategy=iterative-ilp")
  target_include_directories(gufo_qwen_image_21_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
    ${GUFO_EXTENSION_BUILD_ROOT}/hip
  )
endif()
