# YuE2 music generation extension.

gufo_extension_target(
  NAME gufo_yue2
  SOURCES
    config.cpp
    noise.cpp
    protocol.cpp
    safetensors.cpp
    sampler.cpp
    tokenizer.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
    ICU::uc
)

if(TARGET gufo_yue2)
  target_include_directories(gufo_yue2 PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

gufo_extension_target(
  NAME gufo_yue2_hip
  SOURCES
    hip/backbone.cpp
    hip/vae.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_yue2
    gufo_qwen_kernels
    gufo_qwen_eager
    hip::host
    roc::hipblas
)

if(TARGET gufo_yue2_hip)
  gufo_extension_hip_sources(gufo_yue2_hip
    SOURCES
      hip/kernels.hip
  )
  target_compile_definitions(gufo_yue2_hip PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_yue2_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

if(BUILD_TESTING AND TARGET gufo_yue2)
  add_executable(yue2_cpu_test ${CMAKE_CURRENT_LIST_DIR}/tests/cpu_test.cpp)
  target_link_libraries(yue2_cpu_test PRIVATE gufo_yue2)
  add_test(NAME yue2_cpu_test COMMAND yue2_cpu_test)
  set_tests_properties(yue2_cpu_test PROPERTIES
    LABELS "cpu;models;yue2;quality")
endif()

if(BUILD_TESTING AND TARGET gufo_yue2_hip)
  add_executable(yue2_hip_oracle_test
    ${CMAKE_CURRENT_LIST_DIR}/tests/hip_oracle_test.cpp)
  target_link_libraries(yue2_hip_oracle_test PRIVATE gufo_yue2 gufo_yue2_hip
    hip::host)
  add_test(NAME yue2_hip_oracle_test COMMAND yue2_hip_oracle_test)
  set_tests_properties(yue2_hip_oracle_test PROPERTIES
    LABELS "gpu;models;yue2;quality;external-model;slow"
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE)
endif()
