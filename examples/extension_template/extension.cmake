gufo_extension_target(
  NAME gufo_template_model
  SOURCES
    src/template_model.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
)

# Source extensions own installation of their additional data and notices.
install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/template_model)
