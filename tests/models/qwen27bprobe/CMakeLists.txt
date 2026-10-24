if (ENGINE_ENABLE_HIP)
  add_executable(qwen27bprobe_layer_probe
    ${CMAKE_CURRENT_SOURCE_DIR}/layer_probe.cpp)
  target_link_libraries(qwen27bprobe_layer_probe PRIVATE gufo_core)
endif()
