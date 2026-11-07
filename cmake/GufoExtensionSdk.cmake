# The extension SDK: what `gufo extension install` needs to build a module for
# this exact gufo build, without its source tree. It is staged in the build
# tree (`<build>/sdk`) and installed beside the executable
# (`<prefix>/lib/gufo/sdk`).
include_guard(GLOBAL)

# Plain copies: configure_file would add each source as a configure dependency
# under a second spelling of paths the extension scripts already depend on.
function(_gufo_sdk_copy source destination)
  get_filename_component(directory "${destination}" DIRECTORY)
  file(MAKE_DIRECTORY "${directory}")
  file(COPY_FILE "${source}" "${destination}" ONLY_IF_DIFFERENT)
endfunction()

function(_gufo_sdk_cache out)
  set(text "")
  foreach(variable IN LISTS ARGN)
    string(REPLACE "\\" "\\\\" value "${${variable}}")
    string(REPLACE "\"" "\\\"" value "${value}")
    string(APPEND text "set(${variable} \"${value}\" CACHE STRING \"\" FORCE)\n")
  endforeach()
  set(${out} "${text}" PARENT_SCOPE)
endfunction()

# Stage the SDK for the executable `target`. Call after HIP and the extension
# build identity are configured.
function(gufo_extension_sdk target)
  set(sdk "${CMAKE_BINARY_DIR}/sdk")
  set(root "${PROJECT_SOURCE_DIR}")
  _gufo_sdk_copy("${root}/cmake/sdk/CMakeLists.txt" "${sdk}/CMakeLists.txt")
  foreach(file GufoExtension.cmake GufoExtensionDiscovery.cmake
      GufoExtensionVersion.cmake GufoExtensionVersion.hpp.in)
    _gufo_sdk_copy("${root}/cmake/${file}" "${sdk}/cmake/${file}")
  endforeach()
  # Official selection is fixed in the host; modules are never part of it.
  file(WRITE "${sdk}/extensions.lock" "{\"schema\": 1, \"extensions\": []}\n")
  _gufo_sdk_copy("${root}/src/core/hip/libstdcxx_noinline.hpp"
    "${sdk}/support/libstdcxx_noinline.hpp")
  file(GLOB_RECURSE api_files RELATIVE "${root}"
    "${root}/include/gufo/ext/*")
  set(staged)
  foreach(file IN LISTS api_files)
    _gufo_sdk_copy("${root}/${file}" "${sdk}/${file}")
    list(APPEND staged "${sdk}/${file}")
  endforeach()
  file(STRINGS "${root}/include/gufo/ext/SURFACE" headers REGEX "^[^#].+")
  foreach(header IN LISTS headers)
    if(NOT header MATCHES "^include/")
      _gufo_sdk_copy("${root}/${header}" "${sdk}/public-include/${header}")
      list(APPEND staged "${sdk}/public-include/${header}")
    endif()
  endforeach()
  # A removed API header must not stay reachable in a reused build tree.
  file(GLOB_RECURSE previous LIST_DIRECTORIES FALSE "${sdk}/include/*" "${sdk}/public-include/*")
  foreach(file IN LISTS previous)
    if(NOT file IN_LIST staged)
      file(REMOVE "${file}")
    endif()
  endforeach()

  gufo_extension_build_id(build_id)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" config)
  set(GUFO_SDK_BUILD_ID "${build_id}")
  # Nix's CMake searches NIXPKGS_CMAKE_PREFIX_PATH instead of the usual list.
  set(prefix_path ${CMAKE_PREFIX_PATH} $ENV{CMAKE_PREFIX_PATH}
    $ENV{NIXPKGS_CMAKE_PREFIX_PATH})
  if(NOT WIN32)
    string(REPLACE ":" ";" prefix_path "${prefix_path}")
  endif()
  list(REMOVE_DUPLICATES prefix_path)
  set(CMAKE_PREFIX_PATH "${prefix_path}")
  set(GUFO_SDK_PKG_CONFIG_PATH "$ENV{PKG_CONFIG_PATH}")
  set(GUFO_SDK_HIP_DEVICE_LIB_PATH "$ENV{HIP_DEVICE_LIB_PATH}")
  set(variables GUFO_SDK_BUILD_ID GUFO_SDK_PKG_CONFIG_PATH
    GUFO_SDK_HIP_DEVICE_LIB_PATH CMAKE_PREFIX_PATH CMAKE_BUILD_TYPE
    CMAKE_C_COMPILER CMAKE_CXX_COMPILER CMAKE_C_FLAGS CMAKE_C_FLAGS_${config}
    CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_${config} ENGINE_ENABLE_HIP
    GUFO_ENABLE_WARNINGS GUFO_ENABLE_SANITIZERS)
  if(ENGINE_ENABLE_HIP)
    list(APPEND variables CMAKE_HIP_COMPILER CMAKE_HIP_FLAGS
      CMAKE_HIP_FLAGS_${config} CMAKE_HIP_ARCHITECTURES CMAKE_HIP_COMPILER_ROCM_ROOT)
  endif()
  # Pin the dependencies the host found, so the SDK resolves the same ones
  # outside the environment that configured gufo.
  set(dependencies)
  get_cmake_property(cache_variables CACHE_VARIABLES)
  foreach(variable IN LISTS cache_variables)
    if(variable MATCHES "^[A-Za-z0-9_-]+_DIR$" AND NOT variable MATCHES "^(CMAKE|gufo|GUFO)_"
        OR variable MATCHES "^(ICU|OPENSSL|PNG|JPEG|ZLIB|CURL|HIPCUB|ROCPRIM|ROCWMMA)_(INCLUDE_DIR|LIBRARY|.*_LIBRARY|.*_LIBRARY_RELEASE|LIBRARY_RELEASE)$"
        OR variable MATCHES "^(VCPKG_|OpenMP_)")
      get_property(type CACHE "${variable}" PROPERTY TYPE)
      if(NOT type STREQUAL "INTERNAL" AND NOT "${${variable}}" MATCHES "-NOTFOUND$")
        list(APPEND variables "${variable}")
        foreach(item IN LISTS ${variable})
          if(IS_ABSOLUTE "${item}")
            list(APPEND dependencies "${item}")
          endif()
        endforeach()
      endif()
    endif()
  endforeach()
  # A toolchain file (vcpkg's on Windows) stays where the build found it.
  foreach(variable CMAKE_TOOLCHAIN_FILE CMAKE_MSVC_RUNTIME_LIBRARY
      CMAKE_MSVC_DEBUG_INFORMATION_FORMAT CMAKE_POLICY_DEFAULT_CMP0141)
    if(DEFINED ${variable} AND NOT "${${variable}}" STREQUAL "")
      list(APPEND variables ${variable})
    endif()
  endforeach()
  if(CMAKE_TOOLCHAIN_FILE)
    list(APPEND dependencies "${CMAKE_TOOLCHAIN_FILE}")
  endif()
  _gufo_sdk_cache(profile ${variables})
  file(WRITE "${sdk}/profile.cmake"
    "# Generated. The build profile of the gufo executable this SDK belongs to.\n${profile}")

  # What `gufo extension` checks and runs before configuring a module.
  string(JSON sdk_json SET "{}" schema 1)
  string(JSON sdk_json SET "${sdk_json}" build_id "\"${build_id}\"")
  foreach(field cmake generator make_program hip_device_lib_path)
    if(field STREQUAL cmake)
      set(value "${CMAKE_COMMAND}")
    elseif(field STREQUAL generator)
      set(value "${CMAKE_GENERATOR}")
    elseif(field STREQUAL make_program)
      set(value "${CMAKE_MAKE_PROGRAM}")
    else()
      set(value "${GUFO_SDK_HIP_DEVICE_LIB_PATH}")
    endif()
    string(REPLACE "\\" "/" value "${value}")
    string(JSON sdk_json SET "${sdk_json}" ${field} "\"${value}\"")
  endforeach()
  string(JSON sdk_json SET "${sdk_json}" compilers "{}")
  foreach(tool c:CMAKE_C_COMPILER cxx:CMAKE_CXX_COMPILER hip:CMAKE_HIP_COMPILER)
    string(REPLACE ":" ";" pair "${tool}")
    list(GET pair 0 key)
    list(GET pair 1 variable)
    if(DEFINED ${variable} AND NOT "${${variable}}" STREQUAL "")
      string(REPLACE "\\" "/" value "${${variable}}")
      string(JSON sdk_json SET "${sdk_json}" compilers ${key} "\"${value}\"")
    endif()
  endforeach()
  # The environment a toolchain wrapper reads (Nix's hipClang and cc-wrapper
  # take their ROCm and library paths from it). `gufo extension` exports each
  # one that is unset when it runs the SDK.
  string(JSON sdk_json SET "${sdk_json}" environment "{}")
  execute_process(COMMAND "${CMAKE_COMMAND}" -E environment
    OUTPUT_VARIABLE environment_text)
  string(REGEX MATCHALL "(^|\n)[A-Za-z_][A-Za-z0-9_]*=" names "${environment_text}")
  set(environment)
  foreach(name IN LISTS names)
    string(REGEX REPLACE "^\n?([^=]+)=$" "\\1" name "${name}")
    if(name MATCHES "^(HIP_PATH|HSA_PATH|ROCM_PATH|HIP_PLATFORM|HIP_CLANG_PATH|HIP_DEVICE_LIB_PATH|PKG_CONFIG_PATH|NIXPKGS_CMAKE_PREFIX_PATH|NIX_CFLAGS_COMPILE|NIX_LDFLAGS|NIX_HARDENING_ENABLE|NIX_ENFORCE_NO_NATIVE|NIX_(CC|BINTOOLS|PKG_CONFIG)_WRAPPER_TARGET_HOST_.*)$")
      list(APPEND environment "${name}")
    endif()
  endforeach()
  foreach(variable IN LISTS environment)
    if(DEFINED ENV{${variable}})
      set(value "$ENV{${variable}}")
      string(REPLACE "\\" "/" value "${value}")
      string(REPLACE "\"" "\\\"" value "${value}")
      string(JSON sdk_json SET "${sdk_json}" environment ${variable} "\"${value}\"")
    endif()
  endforeach()
  # Paths the profile pins; `gufo extension install` checks they still exist.
  string(JSON sdk_json SET "${sdk_json}" dependencies "[]")
  set(index 0)
  foreach(dependency IN LISTS dependencies)
    string(REPLACE "\\" "/" dependency "${dependency}")
    string(JSON sdk_json SET "${sdk_json}" dependencies ${index} "\"${dependency}\"")
    math(EXPR index "${index}+1")
  endforeach()
  file(WRITE "${sdk}/sdk.json" "${sdk_json}\n")
  _gufo_sdk_copy("${CMAKE_BINARY_DIR}/extension-build-id.txt"
    "${sdk}/extension-build-id.txt")

  # Packages that split the SDK from the runtime install it elsewhere and point
  # gufo at it with GUFO_SDK_DIR.
  set(GUFO_SDK_INSTALL_DIR lib/gufo/sdk CACHE STRING "Extension SDK install directory")
  if(WIN32)
    # Modules link the executable's import library.
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${sdk}/lib"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_LINKER_FILE:${target}>" "${sdk}/lib/$<TARGET_LINKER_FILE_NAME:${target}>"
      VERBATIM)
  endif()
  install(DIRECTORY "${sdk}/" DESTINATION "${GUFO_SDK_INSTALL_DIR}")
endfunction()
