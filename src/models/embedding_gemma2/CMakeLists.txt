# EmbeddingGemma 2 text, image and audio embeddings: model-private BF16 safetensors runtime.

add_library(gufo_embedding_gemma2 STATIC
  audio_processor.cpp
  audio_decoder.cpp
  video_processor.cpp
  batch_scheduler.cpp
  image_processor.cpp
  config.cpp
  loader.cpp
  tensor.cpp
  tokenizer.cpp
  tokenizer_loader.cpp
)

target_include_directories(gufo_embedding_gemma2 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_embedding_gemma2 PUBLIC Threads::Threads
  gufo_platform PkgConfig::SNDFILE PkgConfig::SOXR)

# Video libraries load on the first video request. Linking them into gufo
# adds their codec/network dependency initialization to every service startup.
target_include_directories(gufo_embedding_gemma2 PRIVATE ${AV_INCLUDE_DIRS})
target_compile_options(gufo_embedding_gemma2 PRIVATE ${AV_CFLAGS_OTHER})
set(video_libraries "")
foreach(component IN ITEMS avformat avcodec avutil swscale)
  string(TOUPPER "${component}" upper)
  if(WIN32)
    string(REGEX MATCH "^[0-9]+" major "${AV_lib${component}_VERSION}")
    if(NOT major)
      message(FATAL_ERROR "Missing FFmpeg ${component} runtime version")
    endif()
    list(GET AV_LIBRARY_DIRS 0 runtime_libdir)
    cmake_path(SET runtime NORMALIZE "${runtime_libdir}/../bin/${component}-${major}.dll")
    string(APPEND video_libraries "${component}-${major}.dll\n")
  else()
    find_library(GUFO_${upper}_LIBRARY NAMES ${component}
      HINTS ${AV_LIBRARY_DIRS} REQUIRED)
    # Keep the versioned filename for loading the relocated Linux closure.
    file(REAL_PATH "${GUFO_${upper}_LIBRARY}" runtime)
  endif()
  target_compile_definitions(gufo_embedding_gemma2 PRIVATE
    "GUFO_${upper}_LIBRARY=\"${runtime}\"")
endforeach()
if(WIN32)
  file(GENERATE OUTPUT "${PROJECT_BINARY_DIR}/gufo-video-libraries.txt"
    CONTENT "${video_libraries}")
endif()

# Float32 CPU oracle for the HIP runtime; tests link it explicitly.
add_library(gufo_embedding_gemma2_reference STATIC EXCLUDE_FROM_ALL
  reference/cpu_encoder.cpp
)
target_link_libraries(gufo_embedding_gemma2_reference PUBLIC
  gufo_embedding_gemma2)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_embedding_gemma2 PRIVATE
    hip/audio_runtime.hip
    hip/blas.cpp
    hip/encoder_runtime.cpp
    hip/ops.hip
    hip/vision_ops.hip)
  target_compile_definitions(gufo_embedding_gemma2 PUBLIC ENGINE_ENABLE_HIP=1)
  target_link_libraries(gufo_embedding_gemma2 PUBLIC
    hip::host
    roc::hipblas
    roc::hipblaslt)
endif()
