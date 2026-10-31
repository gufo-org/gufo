# Fish Audio Drama-3 TTS model port — native FP8 (F8_E4M3 + 128x128 scale_inv) MoE + BF16 HIP runtime.

add_library(gufo_fish_drama3 STATIC
  config.cpp
  loader.cpp
  prompt.cpp
  tensor.cpp
  tokenizer.cpp
  hip/backbone_runtime.cpp
  hip/codec_runtime.cpp
  hip/synthesis_runtime.cpp
  hip/tower_runtime.cpp
)

target_include_directories(gufo_fish_drama3 PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(gufo_fish_drama3 PUBLIC
  gufo_core
  ICU::uc
  ICU::i18n
  Threads::Threads)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_fish_drama3 PRIVATE
    hip/backbone_ops.hip
    hip/codec_ops.hip
    hip/tower_ops.hip)
  target_compile_definitions(gufo_fish_drama3 PRIVATE ENGINE_ENABLE_HIP=1)
  target_link_libraries(gufo_fish_drama3 PUBLIC
    hip::host
    roc::hipblas
    roc::rocblas)
endif()
