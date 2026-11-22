# Demand-mapped state memory. Like gufo_cache and gufo_sched, the ledger and
# its host policy build without serving, model or HIP code (tests/mem checks
# that boundary). The page pool and regions are its HIP backend,
# gufo_mem_hip, in src/core/hip/state_*.
add_library(gufo_mem STATIC account.cpp ledger.cpp lendable.cpp
  remap_barrier.cpp shared_rows.cpp state_declaration.cpp)
target_include_directories(gufo_mem PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_mem PUBLIC gufo_platform)
if (BUILD_TESTING)
  target_compile_options(gufo_mem PRIVATE -UNDEBUG)
endif()
