# Qwen3.8-Flash-Next EXL3 2.05 bpw checkpoint: a provider over the
# Flash-Next runtime of the qwen38_flash_next extension.

gufo_extension_target(
  NAME gufo_qwen38_flash_next_exl3_2bpw
  SOURCES
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
)

gufo_extension_target(
  NAME gufo_qwen38_flash_next_exl3_2bpw_hip
  SOURCES
    provider.cpp
  LINK_LIBRARIES
    gufo_qwen38_flash_next_hip
)

if(TARGET gufo_qwen38_flash_next_exl3_2bpw_hip)
  target_compile_definitions(gufo_qwen38_flash_next_exl3_2bpw_hip PRIVATE ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_qwen38_flash_next_exl3_2bpw_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()
