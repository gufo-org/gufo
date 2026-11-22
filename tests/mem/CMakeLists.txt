add_executable(mem_ledger_test ledger_test.cpp)
target_link_libraries(mem_ledger_test PRIVATE gufo_mem)
target_compile_options(mem_ledger_test PRIVATE -UNDEBUG)
add_test(NAME mem_ledger_test COMMAND mem_ledger_test)
set_tests_properties(mem_ledger_test PROPERTIES LABELS "cpu;mem" TIMEOUT 30)
add_executable(mem_state_declaration_test state_declaration_test.cpp)
target_link_libraries(mem_state_declaration_test PRIVATE gufo_mem)
target_compile_options(mem_state_declaration_test PRIVATE -UNDEBUG)
add_test(NAME mem_state_declaration_test COMMAND mem_state_declaration_test)
set_tests_properties(mem_state_declaration_test PROPERTIES LABELS "cpu;mem" TIMEOUT 30)
add_executable(mem_remap_barrier_test remap_barrier_test.cpp)
target_link_libraries(mem_remap_barrier_test PRIVATE gufo_mem Threads::Threads)
target_compile_options(mem_remap_barrier_test PRIVATE -UNDEBUG)
add_test(NAME mem_remap_barrier_test COMMAND mem_remap_barrier_test)
set_tests_properties(mem_remap_barrier_test PROPERTIES LABELS "cpu;mem" TIMEOUT 30)

if (ENGINE_ENABLE_HIP)
  add_executable(mem_region_test region_test.cpp)
  set_source_files_properties(region_test.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(mem_region_test PRIVATE gufo_mem_hip)
  target_compile_options(mem_region_test PRIVATE -UNDEBUG)
  add_test(NAME mem_region_test COMMAND mem_region_test)
  set_tests_properties(mem_region_test PROPERTIES
    LABELS "mem;hip;gpu;gfx1151" TIMEOUT 120)
  add_executable(mem_session_state_test session_state_test.cpp)
  set_source_files_properties(session_state_test.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(mem_session_state_test PRIVATE gufo_mem_hip)
  target_compile_options(mem_session_state_test PRIVATE -UNDEBUG)
  add_test(NAME mem_session_state_test COMMAND mem_session_state_test)
  set_tests_properties(mem_session_state_test PROPERTIES
    LABELS "mem;hip;gpu;gfx1151" TIMEOUT 120)
  add_executable(mem_scratch_arena_test scratch_arena_test.cpp)
  set_source_files_properties(scratch_arena_test.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(mem_scratch_arena_test PRIVATE gufo_mem_hip)
  target_compile_options(mem_scratch_arena_test PRIVATE -UNDEBUG)
  add_test(NAME mem_scratch_arena_test COMMAND mem_scratch_arena_test)
  set_tests_properties(mem_scratch_arena_test PROPERTIES
    LABELS "mem;hip;gpu;gfx1151" TIMEOUT 120)
endif()

if (NOT WIN32)
  file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mem-links-$<CONFIG>.txt CONTENT
    "$<TARGET_GENEX_EVAL:gufo_mem,$<TARGET_PROPERTY:gufo_mem,LINK_LIBRARIES>>\n$<TARGET_GENEX_EVAL:gufo_mem,$<TARGET_PROPERTY:gufo_mem,INTERFACE_LINK_LIBRARIES>>\n")
  # The ledger links only the platform leaves; the HIP backend lives in
  # src/core/hip/state_* and is checked by mem_region_test.
  add_test(NAME mem_boundary_test
    COMMAND ${Python3_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/tests/support/package_boundary_test.py
      ${PROJECT_SOURCE_DIR}/src/mem
      ${CMAKE_CURRENT_BINARY_DIR}/mem-links-$<CONFIG>.txt gufo_platform)
  set_tests_properties(mem_boundary_test PROPERTIES LABELS "cpu;mem" TIMEOUT 30)
endif()

if (ENGINE_ENABLE_HIP)
  add_executable(mem_cache_lending_test cache_lending_test.cpp)
  target_link_libraries(mem_cache_lending_test PRIVATE gufo_cache_hip gufo_mem_hip)
  target_compile_options(mem_cache_lending_test PRIVATE -UNDEBUG)
  add_test(NAME mem_cache_lending_test COMMAND mem_cache_lending_test)
  set_tests_properties(mem_cache_lending_test PROPERTIES
    LABELS "mem;hip;gpu;gfx1151" TIMEOUT 120)
  add_executable(mem_cache_surrender_test cache_surrender_test.cpp)
  target_link_libraries(mem_cache_surrender_test PRIVATE gufo_cache_hip gufo_mem_hip)
  target_compile_options(mem_cache_surrender_test PRIVATE -UNDEBUG)
  add_test(NAME mem_cache_surrender_test COMMAND mem_cache_surrender_test)
  set_tests_properties(mem_cache_surrender_test PROPERTIES
    LABELS "mem;hip;gpu;gfx1151" TIMEOUT 120)
endif()
