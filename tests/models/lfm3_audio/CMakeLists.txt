if (ENGINE_ENABLE_HIP)
  add_executable(lfm3_audio_probe EXCLUDE_FROM_ALL probe.cpp)
  target_link_libraries(lfm3_audio_probe PRIVATE gufo_lfm3_audio)
endif()
