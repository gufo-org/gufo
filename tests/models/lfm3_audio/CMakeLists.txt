if (ENGINE_ENABLE_HIP)
  add_executable(lfm3_audio_test audio_test.cpp)
  target_link_libraries(lfm3_audio_test PRIVATE gufo_lfm3_audio)
  add_test(NAME lfm3_audio.host COMMAND lfm3_audio_test)
  set_tests_properties(lfm3_audio.host PROPERTIES LABELS "lfm3-audio;cpu")
  # Needs weights and reference fixtures; see docs/models/lfm3-audio.
  add_executable(lfm3_audio_probe EXCLUDE_FROM_ALL probe.cpp)
  target_link_libraries(lfm3_audio_probe PRIVATE gufo_lfm3_audio)
endif()
