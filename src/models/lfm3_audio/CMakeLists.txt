if (ENGINE_ENABLE_HIP)
  add_library(gufo_lfm3_audio STATIC weights.cpp audio.cpp model.cpp
    generation.cpp hip/ops.hip)
  target_include_directories(gufo_lfm3_audio PUBLIC ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_lfm3_audio PUBLIC gufo_lfm3 hip::host)
  target_compile_options(gufo_lfm3_audio PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>)
endif()
