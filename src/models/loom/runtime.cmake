# The Loom runtime the loom extension loads: libloom_text_embed
# (loom_text_embed.dll on Windows), Loom's text model layer behind the C API in
# embed/. It drives AMD GPUs through Loom's own driver and needs no ROCm.
#
# A `loom-text-embed` pkg-config module is used when one is found (the Nix
# build provides it, .devops/nix/hrx/loom-runtime.nix). Otherwise the library
# is built here from the hrx-system release that loom-fun packages target;
# HRX's configure step downloads its own pinned dependencies.
#
# Sets GUFO_LOOM_LIBRARY (the library as built) and the target
# gufo_loom_runtime, and installs the library where the runner looks for it:
# beside gufo.exe on Windows, <prefix>/lib/gufo elsewhere.

include_guard(GLOBAL)

# hrx-system v0.2.0 (hrx-loom 0.2.0). Keep in step with
# LOOM_TEXT_EMBED_HRX_LOOM in embed/loom_text_embed.h and
# .devops/nix/hrx/loom-runtime.nix.
set(GUFO_LOOM_HRX_VERSION 0.2.0)
set(GUFO_LOOM_HRX_REVISION 0cd370f2baaa0cfef47cc23007f1a109906f8a52)
set(GUFO_LOOM_HRX_SHA256 2bdf75bc99d530054dc7457ec68df0269f7fd23675115faa361702981cc51fa9)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/embed/loom_text_embed.h" _gufo_loom_sdk
  REGEX "^#define LOOM_TEXT_EMBED_HRX_LOOM ")
if(NOT _gufo_loom_sdk STREQUAL "#define LOOM_TEXT_EMBED_HRX_LOOM \"${GUFO_LOOM_HRX_VERSION}\"")
  message(FATAL_ERROR "loom: embed/loom_text_embed.h names another Loom SDK than hrx-system ${GUFO_LOOM_HRX_VERSION}")
endif()

if(WIN32)
  set(_gufo_loom_name loom_text_embed.dll)
  set(_gufo_loom_destination bin)
else()
  set(_gufo_loom_name libloom_text_embed.so)
  set(_gufo_loom_destination lib/gufo)
endif()

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(GUFO_LOOM_PC QUIET loom-text-embed)
endif()

if(GUFO_LOOM_PC_FOUND)
  find_file(GUFO_LOOM_LIBRARY ${_gufo_loom_name}
    HINTS ${GUFO_LOOM_PC_LIBRARY_DIRS} NO_DEFAULT_PATH REQUIRED)
  add_custom_target(gufo_loom_runtime)
  message(STATUS "loom: Loom runtime from pkg-config: ${GUFO_LOOM_LIBRARY}")
else()
  if(CMAKE_VERSION VERSION_LESS 3.26 OR (WIN32 AND CMAKE_VERSION VERSION_LESS 4.2))
    message(FATAL_ERROR
      "loom: building the Loom runtime needs CMake 3.26 or newer (4.2 on Windows). "
      "Upgrade CMake, provide a loom-text-embed pkg-config module, or leave the "
      "extension out with -DGUFO_EXTENSIONS=<list>.")
  endif()
  find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
  include(ExternalProject)
  # Short paths: HRX's deepest build outputs need about 170 characters.
  set(_gufo_loom_root "${CMAKE_BINARY_DIR}/loom")
  set(GUFO_LOOM_LIBRARY "${_gufo_loom_root}/b/${_gufo_loom_name}")
  set(_gufo_loom_args
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    # Loom's static libraries link into the shared embedding library.
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DIREE_ENABLE_POSITION_INDEPENDENT_CODE=ON
    -DIREE_BUILD_TESTS=OFF
    -DIREE_ENABLE_WERROR_FLAG=OFF
    -DLIBHRX_BUILD=OFF
    -DIREE_HAL_DRIVER_AMD_XDNA=OFF
    -DLOOM_TARGET_ALL_AMD_GPU=ON
    -DIREE_CMAKE_TRY_FILE=${CMAKE_CURRENT_LIST_DIR}/embed/loom_text_embed.cmake)
  if(CMAKE_C_COMPILER_LAUNCHER)
    list(APPEND _gufo_loom_args
      -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
      -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER})
  endif()
  ExternalProject_Add(gufo_loom_runtime
    URL https://github.com/ROCm/hrx-system/archive/${GUFO_LOOM_HRX_REVISION}.tar.gz
    URL_HASH SHA256=${GUFO_LOOM_HRX_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    PREFIX "${_gufo_loom_root}"
    SOURCE_DIR "${_gufo_loom_root}/s"
    BINARY_DIR "${_gufo_loom_root}/b"
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS ${_gufo_loom_args}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target loom_text_embed
    BUILD_BYPRODUCTS "${GUFO_LOOM_LIBRARY}"
    INSTALL_COMMAND ""
    USES_TERMINAL_CONFIGURE ON
    USES_TERMINAL_BUILD ON)
  # The embedding library's sources are this extension's: rebuild on change.
  ExternalProject_Add_StepDependencies(gufo_loom_runtime build
    ${CMAKE_CURRENT_LIST_DIR}/embed/loom_text_embed.c
    ${CMAKE_CURRENT_LIST_DIR}/embed/loom_text_embed.h)
  ExternalProject_Add_StepDependencies(gufo_loom_runtime configure
    ${CMAKE_CURRENT_LIST_DIR}/embed/loom_text_embed.cmake)
  message(STATUS "loom: building the Loom runtime from hrx-system ${GUFO_LOOM_HRX_REVISION}")
endif()

if(WIN32)
  # Windows finds the library beside gufo.exe, also in the build tree.
  add_custom_command(TARGET gufo_loom_runtime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GUFO_LOOM_LIBRARY}" "${CMAKE_BINARY_DIR}"
    VERBATIM)
endif()
install(FILES "${GUFO_LOOM_LIBRARY}" DESTINATION ${_gufo_loom_destination})
if(GUFO_LOOM_PC_FOUND)
  set(_gufo_loom_license "${GUFO_LOOM_PC_PREFIX}/share/licenses/loom-runtime/LICENSE")
else()
  set(_gufo_loom_license "${_gufo_loom_root}/s/LICENSE")
endif()
install(FILES "${_gufo_loom_license}" OPTIONAL
  DESTINATION share/licenses/gufo/third-party RENAME LICENSE.hrx-system)
