# Synthetic metadata/targets live in a separate CMake project: real extensions
# may use the same names, and BUILD_TESTING must not alter production metadata.
if(NOT GUFO_EXTENSION_FIXTURE)
  include(ExternalProject)
  set(fixture_build "${CMAKE_CURRENT_BINARY_DIR}/isolated-fixture")
  ExternalProject_Add(extension_fixture_build
    SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/fixture"
    BINARY_DIR "${fixture_build}"
    CMAKE_ARGS
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DENGINE_ENABLE_HIP=${ENGINE_ENABLE_HIP}"
      "-DBUILD_TESTING=ON"
      "-DGUFO_ENABLE_WARNINGS=${GUFO_ENABLE_WARNINGS}"
      "-DGUFO_ENABLE_SANITIZERS=${GUFO_ENABLE_SANITIZERS}"
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target
      extension_runtime_test extension_role_test
      ext_runtime_neutral_contracts_test
    INSTALL_COMMAND ""
    BUILD_ALWAYS TRUE)
  add_custom_target(extension_runtime_test DEPENDS extension_fixture_build)
  add_custom_target(extension_role_test DEPENDS extension_fixture_build)
  add_custom_target(ext_runtime_neutral_contracts_test
    DEPENDS extension_fixture_build)
  add_test(NAME extension_runtime_test COMMAND "${fixture_build}/extensions/extension_runtime_test${CMAKE_EXECUTABLE_SUFFIX}")
  add_test(NAME extension_role_test COMMAND "${fixture_build}/extensions/extension_role_test${CMAKE_EXECUTABLE_SUFFIX}")
  add_test(NAME ext_runtime_neutral_contracts_test
    COMMAND "${fixture_build}/extensions/ext_runtime_neutral_contracts_test${CMAKE_EXECUTABLE_SUFFIX}")
  set_tests_properties(extension_runtime_test extension_role_test
    ext_runtime_neutral_contracts_test PROPERTIES LABELS "cpu;extensions")
  if(Python3_EXECUTABLE)
    add_test(NAME extension_cli_provenance_test COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_LIST_DIR}/cli_provenance_test.py $<TARGET_FILE:gufo>)
    set_tests_properties(extension_cli_provenance_test PROPERTIES LABELS "cpu;extensions")
    add_test(NAME extension_discovery_test COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_LIST_DIR}/discovery_test.py)
    # Nested projects configure and build repeatedly; Windows needs the headroom.
    # They use the build's own compiler instead of whichever one CMake finds.
    set_tests_properties(extension_discovery_test PROPERTIES LABELS "cpu;extensions" TIMEOUT 900
      ENVIRONMENT "CXX=${CMAKE_CXX_COMPILER}")
    add_test(NAME extension_api_policy_test COMMAND ${Python3_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/tests/tools/check_extension_api_test.py)
    add_test(NAME extension_api_surface COMMAND ${Python3_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/tools/ci/check-extension-api.py)
    set_tests_properties(extension_api_policy_test extension_api_surface PROPERTIES LABELS "cpu;extensions")
    add_test(NAME extension_metadata_test COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_LIST_DIR}/metadata_test.py)
    set_tests_properties(extension_metadata_test PROPERTIES LABELS "cpu;extensions" TIMEOUT 300
      ENVIRONMENT "CXX=${CMAKE_CXX_COMPILER}")
    if(NOT WIN32)
      add_test(NAME extension_schema_test COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_LIST_DIR}/metadata_test.py --schema)
      set_tests_properties(extension_schema_test PROPERTIES LABELS "cpu;extensions" TIMEOUT 30)
    endif()
  endif()
  return()
endif()
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
# Card 09 (RFC step 6): the neutral runtime contracts must stand on their own.
# gufo_ext_runtime is the only library on this link line, so a continuation
# base, host budget or device probe that still needed serving would fail here.
add_executable(ext_runtime_neutral_contracts_test
  ext_runtime_neutral_contracts_test.cpp)
target_link_libraries(ext_runtime_neutral_contracts_test PRIVATE gufo_ext_runtime)
add_test(NAME ext_runtime_neutral_contracts_test
  COMMAND ext_runtime_neutral_contracts_test)
set_tests_properties(ext_runtime_neutral_contracts_test PROPERTIES
  LABELS "cpu;extensions")

add_executable(extension_runtime_test runtime_test.cpp)
target_compile_definitions(extension_runtime_test PRIVATE
  TEST_PROVIDER_HIP=$<BOOL:${ENGINE_ENABLE_HIP}>)
gufo_target_extensions(extension_runtime_test EXTENSIONS beta base alpha)
add_test(NAME extension_runtime_test COMMAND extension_runtime_test)
set_tests_properties(extension_runtime_test PROPERTIES LABELS "cpu;extensions")
