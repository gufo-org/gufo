find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(cargo_target_dir "${CMAKE_CURRENT_BINARY_DIR}/cargo")
set(png_decode_library
  "${cargo_target_dir}/release/${CMAKE_STATIC_LIBRARY_PREFIX}gufo_png_decode${CMAKE_STATIC_LIBRARY_SUFFIX}")
file(GLOB_RECURSE png_decode_sources CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.rs")

add_custom_command(
  OUTPUT "${png_decode_library}"
  COMMAND "${CMAKE_COMMAND}" -E env "CARGO_TARGET_DIR=${cargo_target_dir}"
    "${CARGO_EXECUTABLE}" build --release --locked
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS ${png_decode_sources}
    "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
    "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock"
  COMMENT "Building the Rust PNG decoder"
  VERBATIM)
add_custom_target(gufo_png_decode_cargo DEPENDS "${png_decode_library}")

add_library(gufo_png_decode STATIC IMPORTED GLOBAL)
set_target_properties(gufo_png_decode PROPERTIES
  IMPORTED_LOCATION "${png_decode_library}"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_dependencies(gufo_png_decode gufo_png_decode_cargo)

# The system libraries Rust's standard library needs when linked statically,
# as `cargo rustc --release -- --print native-static-libs` reports them for
# x86_64-unknown-linux-gnu and x86_64-pc-windows-msvc. Rerun it when the
# Rust toolchain or the png crate changes.
if(WIN32)
  target_link_libraries(gufo_png_decode INTERFACE
    kernel32 ntdll userenv ws2_32 dbghelp)
else()
  target_link_libraries(gufo_png_decode INTERFACE
    gcc_s util rt pthread m dl c)
endif()
