# Logging has one process-wide implementation, with an explicit platform edge.
add_library(gufo_logging STATIC
  ${PROJECT_SOURCE_DIR}/src/core/logging/logging.cpp
  ${PROJECT_SOURCE_DIR}/src/core/logging/trace.cpp)
target_include_directories(gufo_logging PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_logging PUBLIC cxx_std_20)
target_link_libraries(gufo_logging PUBLIC gufo_platform)

add_library(gufo_ext_runtime STATIC runtime.cpp options.cpp host_budget.cpp
  device_probe.cpp text.cpp)
# Forwarding public headers still use repository-relative includes. Publish
# only their audited transitive closure, never the repository root.
set(public_include "${CMAKE_CURRENT_BINARY_DIR}/public-include")
file(STRINGS "${PROJECT_SOURCE_DIR}/include/gufo/ext/SURFACE" public_headers
  REGEX "^[^#].+")
set(published_headers "${public_include}/gufo/ext/detail/api_version.hpp")
foreach(header IN LISTS public_headers)
  if(header MATCHES "^include/")
    continue()
  endif()
  set(destination "${header}")
  list(APPEND published_headers "${public_include}/${destination}")
  configure_file("${PROJECT_SOURCE_DIR}/${header}"
    "${public_include}/${destination}" COPYONLY)
endforeach()
# A removed API header must not remain reachable in a reused build tree.
file(GLOB_RECURSE previous_headers LIST_DIRECTORIES FALSE "${public_include}/*")
foreach(header IN LISTS previous_headers)
  if(NOT header IN_LIST published_headers)
    file(REMOVE "${header}")
  endif()
endforeach()
include(${PROJECT_SOURCE_DIR}/cmake/GufoExtensionVersion.cmake)
configure_file("${PROJECT_SOURCE_DIR}/cmake/GufoExtensionVersion.hpp.in"
  "${public_include}/gufo/ext/detail/api_version.hpp" @ONLY)
target_include_directories(gufo_ext_runtime
  PUBLIC "${PROJECT_SOURCE_DIR}/include" "${public_include}"
  PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include")
target_compile_features(gufo_ext_runtime PUBLIC cxx_std_20)
target_link_libraries(gufo_ext_runtime PUBLIC gufo_logging)
# Only the probe implementation touches HIP; `device_probe.hpp` exposes no
# device types, so consumers need neither the define nor a HIP include path.
# The isolated extension fixture project compiles this directory without the
# HIP toolchain, so link only where the imported target actually exists.
if (ENGINE_ENABLE_HIP AND TARGET hip::host)
  target_compile_definitions(gufo_ext_runtime PRIVATE ENGINE_ENABLE_HIP=1)
  target_link_libraries(gufo_ext_runtime PRIVATE hip::host)
endif()
