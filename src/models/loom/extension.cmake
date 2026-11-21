# Loom model packages (https://huggingface.co/loom-fun) as Gufo text models,
# on the package's own Loom kernels through libloom_text_embed, which the
# runner loads when the first package opens (loom_api.hpp).

gufo_extension_target(
  NAME gufo_loom
  SOURCES
    loom_runner.cpp
    loom_chat.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
    gufo_cache
    gufo_platform
)

if(TARGET gufo_loom)
  target_include_directories(gufo_loom PRIVATE
    ${GUFO_EXTENSION_BUILD_ROOT}
    ${GUFO_EXTENSION_BUILD_ROOT}/embed
  )
  include(${GUFO_EXTENSION_BUILD_ROOT}/runtime.cmake)
  add_dependencies(gufo_loom gufo_loom_runtime)
  target_compile_definitions(gufo_loom PRIVATE
    "GUFO_LOOM_BUILD_LIBRARY=\"${GUFO_LOOM_LIBRARY}\"")

  install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
    DESTINATION share/licenses/gufo/extensions/loom)
endif()
