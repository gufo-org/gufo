add_library(gufo_cache_fake STATIC fake_adapter.cpp)
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
