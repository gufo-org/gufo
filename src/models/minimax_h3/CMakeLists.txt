# MiniMax H3 model library: video/audio generation runtime, tokenizer, media
# demux and the ROCm kernels.
#
# Card 08 (RFC step 5) moved these sources out of gufo_core so that excluding
# H3 cannot break an unrelated core tool. This directory stays the model's home
# until Card 18 turns it into an extension with its own extension.json
# manifest, so the library here is a plain target, and `library.json` records
# its ownership of the H3 tests and tools.
#
# The root build adds this directory after the warning/sanitizer block: as
# gufo_core sources these translation units compile with `-Wall -Wextra
# -Wpedantic` and the sanitizer flags, and a model directory added before that
# block would silently drop both.
add_library(gufo_minimax_h3 STATIC
  audio_vae.cpp
  denoiser.cpp
  dit.cpp
  generation.cpp
  json.cpp
  "$<$<PLATFORM_ID:Windows>:media_windows.cpp>"
  "$<$<NOT:$<PLATFORM_ID:Windows>>:media_posix.cpp>"
  prompt_encoder.cpp
  runtime.cpp
  sampling.cpp
  sha256.cpp
  tokenizer.cpp
  video_vae.cpp)

target_include_directories(gufo_minimax_h3 PUBLIC ${CMAKE_SOURCE_DIR})
# Media demux and mux shell out to the configure-time executables.
target_compile_definitions(gufo_minimax_h3 PRIVATE
  GUFO_FFMPEG_EXECUTABLE="${GUFO_FFMPEG_EXECUTABLE}"
  GUFO_FFPROBE_EXECUTABLE="${GUFO_FFPROBE_EXECUTABLE}")
target_link_libraries(gufo_minimax_h3 PUBLIC gufo_core)

if (ENGINE_ENABLE_HIP)
  set_property(SOURCE dit.hip APPEND PROPERTY COMPILE_OPTIONS -O2)
  target_sources(gufo_minimax_h3 PRIVATE
    audio_vae.hip
    denoiser.hip
    dit.hip
    dense.hip
    runtime_hip.cpp
    prompt_encoder.hip
    sampling.hip
    video_vae.hip
    attention.hip)
  target_include_directories(gufo_minimax_h3 SYSTEM PRIVATE
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR})
endif()
