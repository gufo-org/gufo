# libloom_text_embed.so (loom_text_embed.dll on Windows): a Loom package's text model layer behind a C API,
# for serving processes that own their own scheduling and caching. HRX's CMake
# includes this file once its targets exist (-DIREE_CMAKE_TRY_FILE=<this file>);
# the library builds in the top of the HRX build tree.
if(NOT (LOOM_ARCH_VM AND LOOM_ARTIFACT_VM_BINARY AND LOOM_ROUTE_AMDGPU_HAL))
  message(FATAL_ERROR "loom_text_embed needs Loom's VM and AMD GPU HAL routes")
endif()
add_library(loom_text_embed SHARED "${CMAKE_CURRENT_LIST_DIR}/loom_text_embed.c")
target_include_directories(loom_text_embed PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(loom_text_embed PRIVATE
  iree::base
  iree::base::internal::path
  iree::base::tooling::flags
  iree::tokenizer
  iree::tokenizer::vocab
  iree::tooling::device_util
  loom::runtime::device
  loom::runtime::host
  loom::serve::server::configuration
  loom::serve::server::device_configuration
  loom::serve::service::text::chat
  loom::serve::service::text::model
  loom::serve::service::text::schedule
  loom::serve::service::text::input
  loom::serve::service::text::media_projector
  loom::service::finite::component
  loom::binding::c::target::configured
  loom::binding::c::target::amdgpu::iree_hal
  # loom_text_embed_query_gpu asks the driver for the default GPU.
  amdf::amdf_static
)
set_target_properties(loom_text_embed PROPERTIES
  C_VISIBILITY_PRESET hidden
  OUTPUT_NAME loom_text_embed)
if(WIN32)
  target_sources(loom_text_embed PRIVATE "${CMAKE_CURRENT_LIST_DIR}/exports.def")
else()
  target_link_options(loom_text_embed PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/exports.map"
    "-Wl,--no-undefined")
endif()
