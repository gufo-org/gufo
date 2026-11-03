# EmbeddingGemma 2 text, image and audio embeddings extension.

gufo_extension_target(
  NAME gufo_embedding_gemma2
  SOURCES
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
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_platform
    Threads::Threads
    PkgConfig::SNDFILE
    PkgConfig::SOXR
)

if(TARGET gufo_embedding_gemma2)
  target_include_directories(gufo_embedding_gemma2 PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )

  # Video libraries load on the first video request.
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
    ${GUFO_EXTENSION_BUILD_ROOT}/reference/cpu_encoder.cpp
  )
  target_include_directories(gufo_embedding_gemma2_reference PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
  target_link_libraries(gufo_embedding_gemma2_reference PUBLIC
    gufo_embedding_gemma2
  )
endif()

gufo_extension_target(
  NAME gufo_embedding_gemma2_hip
  SOURCES
    hip/blas.cpp
    hip/encoder_runtime.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_embedding_gemma2
    hip::host
    roc::hipblas
    roc::hipblaslt
)

if(TARGET gufo_embedding_gemma2_hip)
  gufo_extension_hip_sources(gufo_embedding_gemma2_hip
    SOURCES
      hip/audio_runtime.hip
      hip/ops.hip
      hip/vision_ops.hip
  )
  target_include_directories(gufo_embedding_gemma2_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
    ${GUFO_EXTENSION_BUILD_ROOT}/hip
  )
  target_compile_definitions(gufo_embedding_gemma2_hip PUBLIC ENGINE_ENABLE_HIP=1)
endif()
