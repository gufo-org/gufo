add_library(gufo_cache_fake STATIC fake_adapter.cpp)
if (ENGINE_ENABLE_HIP)
  add_executable(cache_backing_test backing_test.cpp)
  target_link_libraries(cache_backing_test PRIVATE gufo_cache_hip)
  target_compile_options(cache_backing_test PRIVATE -UNDEBUG)
  target_link_options(cache_backing_test PRIVATE
    -Wl,--wrap=hipHostMalloc -Wl,--wrap=hipMalloc
    -Wl,--wrap=hipHostRegister -Wl,--wrap=hipMallocManaged
    -Wl,--wrap=hipStreamCreateWithFlags -Wl,--wrap=hipEventCreateWithFlags
    -Wl,--wrap=hipMemcpyAsync -Wl,--wrap=hipEventRecord
    -Wl,--wrap=hipEventQuery -Wl,--wrap=hipEventSynchronize
    -Wl,--wrap=hipStreamSynchronize)
  add_test(NAME cache_backing_test COMMAND cache_backing_test)
  set_tests_properties(cache_backing_test PROPERTIES
    LABELS "hip;gpu;gfx1151;cache" TIMEOUT 60)
endif()
target_link_libraries(gufo_cache_fake PUBLIC gufo_cache)
add_executable(cache_adapter_test adapter_test.cpp)
target_link_libraries(cache_adapter_test PRIVATE gufo_cache_fake)
add_test(NAME cache_adapter_test COMMAND cache_adapter_test)
set_tests_properties(cache_adapter_test PROPERTIES LABELS "cpu;cache" TIMEOUT 30)

add_executable(cache_adapter_lifecycle_test lifecycle_test.cpp)
target_link_libraries(cache_adapter_lifecycle_test PRIVATE gufo_cache_fake)
add_test(NAME cache_adapter_lifecycle_test COMMAND cache_adapter_lifecycle_test)
set_tests_properties(cache_adapter_lifecycle_test PROPERTIES LABELS "cpu;cache" TIMEOUT 30)

add_executable(cache_ledger_test ledger_test.cpp)
target_link_libraries(cache_ledger_test PRIVATE gufo_cache Threads::Threads)
add_test(NAME cache_ledger_test COMMAND cache_ledger_test)
set_tests_properties(cache_ledger_test PROPERTIES LABELS "cpu;cache" TIMEOUT 30)

file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cache-links-$<CONFIG>.txt CONTENT
  "$<TARGET_GENEX_EVAL:gufo_cache,$<TARGET_PROPERTY:gufo_cache,LINK_LIBRARIES>>\n$<TARGET_GENEX_EVAL:gufo_cache,$<TARGET_PROPERTY:gufo_cache,INTERFACE_LINK_LIBRARIES>>\n")

add_test(NAME cache_boundary_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/boundary_test.py
    ${PROJECT_SOURCE_DIR}/src/cache
    ${CMAKE_CURRENT_BINARY_DIR}/cache-links-$<CONFIG>.txt)
set_tests_properties(cache_boundary_test PROPERTIES LABELS "cpu;cache" TIMEOUT 30)

add_executable(cache_checkpoint_test checkpoint_test.cpp)
target_link_libraries(cache_checkpoint_test PRIVATE gufo_cache_fake)
add_test(NAME cache_checkpoint_test COMMAND cache_checkpoint_test)
set_tests_properties(cache_checkpoint_test PROPERTIES LABELS "cpu;cache" TIMEOUT 30)

add_executable(cache_prefix_index_test prefix_index_test.cpp)
target_link_libraries(cache_prefix_index_test PRIVATE gufo_cache)
add_test(NAME cache_prefix_index_test COMMAND cache_prefix_index_test)
set_tests_properties(cache_prefix_index_test PROPERTIES LABELS "cpu;cache" TIMEOUT 30)

add_executable(cache_slot_test slot_test.cpp)
target_link_libraries(cache_slot_test PRIVATE gufo_cache_fake)
add_test(NAME cache_slot_test COMMAND cache_slot_test)
set_tests_properties(cache_slot_test PROPERTIES LABELS "cpu;cache" TIMEOUT 30)

if (GUFO_BUILD_TOOLS)
  add_executable(cache_slot_bench slot_bench.cpp)
  target_link_libraries(cache_slot_bench PRIVATE gufo_cache_fake)
endif()

add_executable(cache_retention_test retention_test.cpp)
target_link_libraries(cache_retention_test PRIVATE gufo_cache_fake)
add_test(NAME cache_retention_test COMMAND cache_retention_test)
set_tests_properties(cache_retention_test PROPERTIES LABELS "cpu;cache" TIMEOUT 30)

if (GUFO_BUILD_TOOLS)
  add_executable(cache_retention_replay retention_replay.cpp)
  target_link_libraries(cache_retention_replay PRIVATE gufo_cache_fake)
endif()

if (GUFO_BUILD_TOOLS)
  add_executable(cache_retention_bench retention_bench.cpp)
  target_link_libraries(cache_retention_bench PRIVATE gufo_cache_fake)
endif()
