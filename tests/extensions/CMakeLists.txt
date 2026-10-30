include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/GufoExtension.cmake)
# Deliberately reverse discovery order: registry ordering is dependency-sorted.
foreach(name beta alpha base)
  gufo_extension(MANIFEST "${CMAKE_CURRENT_LIST_DIR}/fixture/${name}.json")
endforeach()
gufo_extensions_validate()
gufo_extension_target(NAME extension_test_base SOURCES fixture/base.cpp)
gufo_extension_target(NAME extension_role_test SOURCES fixture/role_test.cpp)
add_test(NAME extension_role_test COMMAND extension_role_test)
set_tests_properties(extension_role_test PROPERTIES LABELS "cpu;extensions")
gufo_extension_target(NAME alpha_registration SOURCES fixture/registration.cpp
  LINK_LIBRARIES gufo_ext_runtime)
gufo_extension_target(NAME alpha_cpu SOURCES fixture/cpu.cpp
  LINK_LIBRARIES gufo_ext_runtime extension_test_base)
gufo_extension_target(NAME alpha_hip SOURCES fixture/hip.cpp
  LINK_LIBRARIES gufo_ext_runtime)
gufo_extension_target(NAME beta_registration SOURCES fixture/beta.cpp
  LINK_LIBRARIES gufo_ext_runtime)
add_executable(extension_runtime_test runtime_test.cpp)
target_compile_definitions(extension_runtime_test PRIVATE
  TEST_PROVIDER_HIP=$<BOOL:${ENGINE_ENABLE_HIP}>)
gufo_target_extensions(extension_runtime_test EXTENSIONS beta base alpha)
add_test(NAME extension_runtime_test COMMAND extension_runtime_test)
set_tests_properties(extension_runtime_test PROPERTIES LABELS "cpu;extensions")
if(Python3_EXECUTABLE)
  add_test(NAME extension_metadata_test COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_LIST_DIR}/metadata_test.py)
  set_tests_properties(extension_metadata_test PROPERTIES LABELS "cpu;extensions" TIMEOUT 60)
  if(NOT WIN32)
    add_test(NAME extension_schema_test COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_LIST_DIR}/metadata_test.py --schema)
    set_tests_properties(extension_schema_test PROPERTIES LABELS "cpu;extensions" TIMEOUT 30)
  endif()
endif()
