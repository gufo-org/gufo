# Gufo extension SDK: builds extensions as modules for the one gufo build that
# installed this SDK. `gufo extension install` drives it:
#
#   cmake -S <sdk> -B <build> -C <sdk>/profile.cmake \
#     -DGUFO_EXTENSION_DIRS=<extension source> -DGUFO_EXTENSION_MODULES=<name>
#
# The profile pins the host's compilers, flags and dependency prefixes; the
# configure step fails when the resulting build identity differs from the
# host's, before anything is compiled.
cmake_minimum_required(VERSION 3.27)
if(NOT DEFINED GUFO_SDK_BUILD_ID)
  message(FATAL_ERROR "configure the Gufo extension SDK with -C <sdk>/profile.cmake")
endif()
project(gufo_extension_module LANGUAGES C CXX)
if(WIN32)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN PSAPI_VERSION=2)
endif()
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_TESTING OFF CACHE BOOL "Extension tests build with a Gufo checkout" FORCE)
if(GUFO_SDK_PKG_CONFIG_PATH)
  set(ENV{PKG_CONFIG_PATH} "${GUFO_SDK_PKG_CONFIG_PATH}")
endif()
# Linux libstdc++ must be parsed before HIP defines its __noinline__ macro.
if(NOT WIN32)
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include \"${CMAKE_CURRENT_LIST_DIR}/support/libstdcxx_noinline.hpp\">")
endif()

# The dependencies the gufo build offers extensions. Only the toolchain and the
# libraries behind the public link contracts are required.
find_package(Threads REQUIRED)
find_package(ICU REQUIRED COMPONENTS uc i18n)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
foreach(package CURL PNG JPEG ZLIB OpenMP)
  find_package(${package} QUIET)
endforeach()
# Extension-only libraries come from each extension's manifest `packages`.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(WEBP QUIET IMPORTED_TARGET libwebp)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/GufoExtensionDiscovery.cmake)
gufo_discover_extensions()
if(EXISTS /opt/rocm)
  list(APPEND CMAKE_SYSTEM_PREFIX_PATH /opt/rocm)
endif()
if(ENGINE_ENABLE_HIP)
  enable_language(HIP)
  if(WIN32)
    string(REPLACE "/Fo<OBJECT>" "-o <OBJECT>" CMAKE_HIP_COMPILE_OBJECT
      "${CMAKE_HIP_COMPILE_OBJECT}")
  endif()
  find_package(hip REQUIRED)
  find_package(hipblas REQUIRED)
  find_package(hipblaslt REQUIRED)
  find_package(rocblas REQUIRED)
  find_package(rocm-core CONFIG QUIET)
  find_path(HIPCUB_INCLUDE_DIR hipcub/hipcub.hpp REQUIRED)
  find_path(ROCPRIM_INCLUDE_DIR rocprim/rocprim.hpp REQUIRED)
  find_path(ROCWMMA_INCLUDE_DIR rocwmma/rocwmma.hpp REQUIRED)
  # Outside a development shell nothing exports the device library path.
  if(GUFO_SDK_HIP_DEVICE_LIB_PATH AND NOT DEFINED ENV{HIP_DEVICE_LIB_PATH})
    add_compile_options(
      "$<$<COMPILE_LANGUAGE:HIP>:--rocm-device-lib-path=${GUFO_SDK_HIP_DEVICE_LIB_PATH}>")
  endif()
endif()
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # As in the gufo build: lld-link skips the clang-cl runtime selection.
  execute_process(COMMAND "${CMAKE_CXX_COMPILER}" /clang:-print-runtime-dir
    OUTPUT_VARIABLE clang_rt_dir OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(EXISTS "${clang_rt_dir}/clang_rt.builtins-x86_64.lib")
    add_link_options("${clang_rt_dir}/clang_rt.builtins-x86_64.lib")
  endif()
endif()
if(GUFO_ENABLE_WARNINGS)
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    add_compile_options(/W4)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
  endif()
endif()
if(GUFO_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# The public link contracts carry headers only: the loading gufo executable
# supplies every implementation.
set(generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
include(${CMAKE_CURRENT_LIST_DIR}/cmake/GufoExtensionVersion.cmake)
configure_file("${CMAKE_CURRENT_LIST_DIR}/cmake/GufoExtensionVersion.hpp.in"
  "${generated}/gufo/ext/detail/api_version.hpp" @ONLY)
add_library(gufo_ext_runtime INTERFACE)
target_include_directories(gufo_ext_runtime INTERFACE
  "${CMAKE_CURRENT_LIST_DIR}/include" "${CMAKE_CURRENT_LIST_DIR}/public-include"
  "${generated}")
target_compile_features(gufo_ext_runtime INTERFACE cxx_std_20)
# `sha256.hpp` in the runtime surface exposes OpenSSL types.
target_link_libraries(gufo_ext_runtime INTERFACE $<COMPILE_ONLY:OpenSSL::Crypto>)
add_library(gufo_ext_core INTERFACE)
target_link_libraries(gufo_ext_core INTERFACE OpenSSL::Crypto)
add_library(gufo_ext_cache INTERFACE)
if(ENGINE_ENABLE_HIP)
  add_library(gufo_ext_hip INTERFACE)
  target_compile_definitions(gufo_ext_hip INTERFACE ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_ext_hip SYSTEM INTERFACE
    ${HIPCUB_INCLUDE_DIR} ${ROCPRIM_INCLUDE_DIR})
  # `hip_utils.hpp` in the public surface includes hipBLAS.
  target_link_libraries(gufo_ext_hip INTERFACE hip::host roc::hipblas)
endif()

if(WIN32)
  # Windows modules import the host services from gufo's import library.
  add_library(gufo_host_import UNKNOWN IMPORTED)
  set_target_properties(gufo_host_import PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/lib/gufo.lib")
  set(GUFO_EXTENSION_MODULE_HOST gufo_host_import)
endif()
if(NOT GUFO_EXTENSION_MODULES)
  message(FATAL_ERROR "GUFO_EXTENSION_MODULES names the extensions to build")
endif()
set(GUFO_SURFACE_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/public-include")
gufo_extension_build_id(build_id)
if(NOT build_id STREQUAL GUFO_SDK_BUILD_ID)
  message(FATAL_ERROR "this toolchain does not match the gufo build that "
    "installed this SDK: build identity ${build_id}, gufo ${GUFO_SDK_BUILD_ID}. "
    "Compare ${CMAKE_BINARY_DIR}/extension-build-id.txt with "
    "${CMAKE_CURRENT_LIST_DIR}/extension-build-id.txt.")
endif()
gufo_build_discovered_extensions()
gufo_extension_modules(EXTENSIONS ${GUFO_EXTENSION_MODULES})
