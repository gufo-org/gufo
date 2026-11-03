# Qwen vision stack: encoder, device-side image input, image prompt expansion
# and the image cache. The prompt/cache translation units are CPU work (they
# only shape placeholders and cache identity), so they build without HIP; the
# encoder and device input need the backend.
add_library(gufo_qwen_vision STATIC
  image_cache.cpp
  prompt.cpp)
target_include_directories(gufo_qwen_vision PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen_vision PUBLIC
  gufo_core
  gufo_qwen_hip_common
  gufo_qwen_text
  gufo_qwen_control_tokens)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_qwen_vision PRIVATE
    encoder.hip
    device_input.hip)
  target_include_directories(gufo_qwen_vision SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR})
  # encoder.hip reaches the shared GEMM launcher, which only exists with HIP;
  # the manifest declares the dependency because the include graph does.
  target_link_libraries(gufo_qwen_vision PUBLIC gufo_qwen_kernels)
endif()
