add_executable(sched_policy_test policy_test.cpp)
target_link_libraries(sched_policy_test PRIVATE gufo_sched)
target_compile_options(sched_policy_test PRIVATE -UNDEBUG)
add_test(NAME sched_policy_test COMMAND sched_policy_test)
set_tests_properties(sched_policy_test PROPERTIES LABELS "cpu;scheduler" TIMEOUT 30)

if (NOT WIN32)
  file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/sched-links-$<CONFIG>.txt CONTENT
    "$<TARGET_GENEX_EVAL:gufo_sched,$<TARGET_PROPERTY:gufo_sched,LINK_LIBRARIES>>\n$<TARGET_GENEX_EVAL:gufo_sched,$<TARGET_PROPERTY:gufo_sched,INTERFACE_LINK_LIBRARIES>>\n")
  # The policy links nothing.
  add_test(NAME sched_boundary_test
    COMMAND ${Python3_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/tests/support/package_boundary_test.py
      ${PROJECT_SOURCE_DIR}/src/sched
      ${CMAKE_CURRENT_BINARY_DIR}/sched-links-$<CONFIG>.txt)
  set_tests_properties(sched_boundary_test PROPERTIES LABELS "cpu;scheduler" TIMEOUT 30)
endif()
