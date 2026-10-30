# Logging has one process-wide implementation, with an explicit platform edge.
add_library(gufo_logging STATIC
  ${PROJECT_SOURCE_DIR}/src/core/logging/logging.cpp
  ${PROJECT_SOURCE_DIR}/src/core/logging/trace.cpp)
target_include_directories(gufo_logging PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_logging PUBLIC cxx_std_20)
target_link_libraries(gufo_logging PUBLIC gufo_platform)

add_library(gufo_ext_runtime STATIC runtime.cpp options.cpp)
target_include_directories(gufo_ext_runtime PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(gufo_ext_runtime PUBLIC cxx_std_20)
target_link_libraries(gufo_ext_runtime PUBLIC gufo_logging)
