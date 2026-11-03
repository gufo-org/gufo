find_program(CARGO_EXECUTABLE cargo REQUIRED)
find_program(RUSTC_EXECUTABLE rustc REQUIRED)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/rust-toolchain.toml")

set(cargo_target_dir "${CMAKE_CURRENT_BINARY_DIR}/cargo")
set(cargo_profile "$<IF:$<CONFIG:Debug>,dev,$<IF:$<CONFIG:RelWithDebInfo>,relwithdebinfo,release>>")
set(cargo_profile_dir "$<IF:$<CONFIG:Debug>,debug,$<IF:$<CONFIG:RelWithDebInfo>,relwithdebinfo,release>>")
set(png_decode_filename "${CMAKE_STATIC_LIBRARY_PREFIX}gufo_png_decode${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(png_decode_library "${cargo_target_dir}/${cargo_profile_dir}/${png_decode_filename}")
file(GLOB_RECURSE png_decode_sources CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.rs")

# Let Cargo check compiler/environment changes even when the archive exists.
# Cargo's own dependency tracking keeps unchanged builds inexpensive.
add_custom_target(gufo_png_decode_cargo
  COMMAND "${CMAKE_COMMAND}" -E env "CARGO_TARGET_DIR=${cargo_target_dir}"
    "${CARGO_EXECUTABLE}" build --profile "${cargo_profile}" --locked
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  BYPRODUCTS "${png_decode_library}"
  DEPENDS ${png_decode_sources}
    "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
    "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock"
    "${PROJECT_SOURCE_DIR}/rust-toolchain.toml"
  COMMENT "Building the Rust PNG decoder (${cargo_profile})"
  VERBATIM)

add_library(gufo_png_decode STATIC IMPORTED GLOBAL)
set_target_properties(gufo_png_decode PROPERTIES
  IMPORTED_LOCATION "${cargo_target_dir}/release/${png_decode_filename}"
  IMPORTED_LOCATION_DEBUG "${cargo_target_dir}/debug/${png_decode_filename}"
  IMPORTED_LOCATION_RELWITHDEBINFO "${cargo_target_dir}/relwithdebinfo/${png_decode_filename}"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_dependencies(gufo_png_decode gufo_png_decode_cargo)

# Keep these explicit for CMake's link-graph classification. Check them against
# the actual compiler so toolchain upgrades cannot silently stale the list.
if(WIN32)
  set(png_decode_native_libs kernel32 ntdll userenv ws2_32 dbghelp)
else()
  set(png_decode_native_libs gcc_s util rt pthread m dl c)
endif()

# A tiny std-using staticlib reports the same standard-library requirements as
# the decoder. Probe from the source directory so rustup uses our toolchain file.
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
set(native_probe_names "")
foreach(lib IN LISTS native_probe_libs)
  string(REGEX REPLACE "^-l" "" lib "${lib}")
  string(REGEX REPLACE "\\.lib$" "" lib "${lib}")
  list(APPEND native_probe_names "${lib}")
endforeach()
list(REMOVE_DUPLICATES native_probe_names)
list(SORT native_probe_names)
set(expected_native_libs ${png_decode_native_libs})
if(WIN32)
  # Rust embeds this directive in its objects; MSVC supplies the dynamic CRT.
  list(APPEND expected_native_libs "/defaultlib:msvcrt")
endif()
list(SORT expected_native_libs)
if(NOT native_probe_names STREQUAL expected_native_libs)
  message(FATAL_ERROR
    "Rust native libraries changed: expected [${expected_native_libs}], got [${native_probe_names}]. Update the decoder linkage and cmake/TargetGraph.cmake.")
endif()
target_link_libraries(gufo_png_decode INTERFACE ${png_decode_native_libs})
