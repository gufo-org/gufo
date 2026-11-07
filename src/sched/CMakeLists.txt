# Scheduling policy for text serving. Like gufo_cache, it builds without
# serving, model or HIP code; tests/sched checks that boundary.
add_library(gufo_sched STATIC policy.cpp)
target_include_directories(gufo_sched PUBLIC ${PROJECT_SOURCE_DIR})
if (BUILD_TESTING)
  target_compile_options(gufo_sched PRIVATE -UNDEBUG)
endif()
