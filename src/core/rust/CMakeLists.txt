find_program(CARGO_EXECUTABLE cargo REQUIRED)
find_program(RUSTC_EXECUTABLE rustc REQUIRED)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/rust-toolchain.toml")

set(cargo_target_dir "${CMAKE_CURRENT_BINARY_DIR}/cargo")
# Test builds keep debug assertions and overflow checks, like their C++
# assertions; shipped builds (BUILD_TESTING=OFF) use the unchecked profiles.
if(BUILD_TESTING)
  set(cargo_optimized_profile checked)
else()
  set(cargo_optimized_profile relwithdebinfo)
endif()
set(cargo_profile "$<IF:$<CONFIG:Debug>,dev,$<IF:$<CONFIG:RelWithDebInfo>,${cargo_optimized_profile},release>>")
set(cargo_profile_dir "$<IF:$<CONFIG:Debug>,debug,$<IF:$<CONFIG:RelWithDebInfo>,${cargo_optimized_profile},release>>")
set(rust_filename "${CMAKE_STATIC_LIBRARY_PREFIX}gufo_rust${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(rust_library "${cargo_target_dir}/${cargo_profile_dir}/${rust_filename}")

# A custom target runs on every build, so Cargo, not CMake, decides what to
# rebuild from sources, manifests and the toolchain. Unchanged builds stay cheap.
add_custom_target(gufo_rust_cargo
  COMMAND "${CMAKE_COMMAND}" -E env "CARGO_TARGET_DIR=${cargo_target_dir}"
    "RUSTC=${RUSTC_EXECUTABLE}"
    "${CARGO_EXECUTABLE}" build --profile "${cargo_profile}" --locked
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  BYPRODUCTS "${rust_library}"
  COMMENT "Building the Rust components (${cargo_profile})"
  VERBATIM)

add_library(gufo_rust STATIC IMPORTED GLOBAL)
set_target_properties(gufo_rust PROPERTIES
  IMPORTED_LOCATION "${cargo_target_dir}/release/${rust_filename}"
  IMPORTED_LOCATION_DEBUG "${cargo_target_dir}/debug/${rust_filename}"
  IMPORTED_LOCATION_RELWITHDEBINFO "${cargo_target_dir}/${cargo_optimized_profile}/${rust_filename}"
  INTERFACE_INCLUDE_DIRECTORIES
    "${PROJECT_SOURCE_DIR}/src/core/png_decode/include;${PROJECT_SOURCE_DIR}/src/core/wav_decode/include")
add_dependencies(gufo_rust gufo_rust_cargo)

# A tiny std-using staticlib reports the same standard-library requirements as
# the components. Probe from the source directory so rustup uses our toolchain
# file, with the same rustc that Cargo builds the components with.
set(native_probe "${CMAKE_CURRENT_BINARY_DIR}/native-libs")
file(WRITE "${native_probe}.rs" "pub fn probe() { let _ = std::env::args(); }\n")
execute_process(
  COMMAND "${RUSTC_EXECUTABLE}" --crate-name gufo_native_libs
    --crate-type staticlib --print native-static-libs
    "${native_probe}.rs" -o "${native_probe}${CMAKE_STATIC_LIBRARY_SUFFIX}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  RESULT_VARIABLE native_probe_status
  OUTPUT_VARIABLE native_probe_stdout
  ERROR_VARIABLE native_probe_stderr)
file(REMOVE "${native_probe}${CMAKE_STATIC_LIBRARY_SUFFIX}")
if(NOT native_probe_status EQUAL 0)
  message(FATAL_ERROR "Rust native-library probe failed: ${native_probe_stderr}")
endif()
string(REGEX MATCH "native-static-libs: ([^\r\n]+)" native_probe_match
  "${native_probe_stdout}${native_probe_stderr}")
if(NOT native_probe_match)
  message(FATAL_ERROR "rustc did not report native-static-libs")
endif()
separate_arguments(native_probe_libs NATIVE_COMMAND "${CMAKE_MATCH_1}")
set(rust_native_libs "")
foreach(lib IN LISTS native_probe_libs)
  # Linker directives such as /defaultlib:msvcrt are embedded in Rust's
  # objects; MSVC supplies the dynamic CRT they name.
  if(lib MATCHES "^[-/]" AND NOT lib MATCHES "^-l")
    continue()
  endif()
  string(REGEX REPLACE "^-l" "" lib "${lib}")
  string(REGEX REPLACE "\\.lib$" "" lib "${lib}")
  list(APPEND rust_native_libs "${lib}")
endforeach()
list(REMOVE_DUPLICATES rust_native_libs)
if(NOT rust_native_libs)
  message(FATAL_ERROR "rustc reported no native-static-libs")
endif()
# cmake/TargetGraph.cmake classifies these as system libraries.
set(GUFO_RUST_LIBS ${rust_native_libs} PARENT_SCOPE)
target_link_libraries(gufo_rust INTERFACE ${rust_native_libs})
