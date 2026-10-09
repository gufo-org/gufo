add_library(gufo_cache_fake STATIC fake_adapter.cpp)
target_link_libraries(gufo_cache_fake PUBLIC gufo_cache)
add_executable(cache_adapter_test adapter_test.cpp)
target_link_libraries(cache_adapter_test PRIVATE gufo_cache_fake)
add_test(NAME cache_adapter_test COMMAND cache_adapter_test)
set_tests_properties(cache_adapter_test PROPERTIES LABELS "cpu;cache" TIMEOUT 30)

add_test(NAME cache_boundary_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/boundary_test.py
    ${PROJECT_SOURCE_DIR}/src/cache)
set_tests_properties(cache_boundary_test PROPERTIES LABELS "cpu;cache" TIMEOUT 30)
