# NPU designs (IRON, src/core/npu/kernels/<name>/<name>.py), listed in
# GufoNpuDesignList.cmake, compiled at build time when an mlir-aie Python is
# available: GUFO_IRON_PYTHON (the dev shell sets it; on Windows the venv
# C:\iron from tools/windows/iron-requirements.txt is found by default), with
# Peano from PEANO_INSTALL_DIR (default: the llvm-aie wheel in that Python).
# Without one, designs and their device tests are skipped; runtimes then load
# designs from directories given at run time.
# GUFO_NPU_DESIGNS_REQUIRED=ON (release builds) makes a missing IRON an error.
# GUFO_NPU_PREBUILT_DESIGNS=<dir> installs designs compiled elsewhere
# (<dir>/<name>/, e.g. the Nix npu-designs derivation) instead.
include_guard(GLOBAL)

set(GUFO_NPU_KERNELS "${CMAKE_CURRENT_LIST_DIR}/../src/core/npu/kernels")
get_filename_component(GUFO_NPU_KERNELS "${GUFO_NPU_KERNELS}" ABSOLUTE)

set(GUFO_NPU_PREBUILT_DESIGNS "" CACHE PATH
  "Directory of compiled NPU designs (<name>/ each) to install instead of compiling")
set(_gufo_iron_default "$ENV{GUFO_IRON_PYTHON}")
if(NOT _gufo_iron_default AND WIN32 AND EXISTS "C:/iron/Scripts/python.exe")
  set(_gufo_iron_default "C:/iron/Scripts/python.exe")
endif()
set(GUFO_IRON_PYTHON "${_gufo_iron_default}" CACHE FILEPATH
  "Python with mlir-aie (IRON) for compiling NPU designs; empty skips them")
option(GUFO_NPU_DESIGNS_REQUIRED "Fail configure when the NPU designs cannot be compiled" OFF)
# IRON caches compiled designs and kernel objects here, keyed on their inputs;
# CI keeps it across runs so unchanged designs are not recompiled. Default:
# the build tree (sandboxed builds have no writable home).
set(GUFO_NPU_CACHE_HOME "${CMAKE_BINARY_DIR}/npu/.cache" CACHE PATH
  "IRON's cache of compiled NPU designs (NPU_CACHE_HOME)")

# Where this build's designs live; the runtime falls back to it uninstalled.
set(GUFO_NPU_DESIGNS_AVAILABLE OFF)
if(GUFO_NPU_PREBUILT_DESIGNS)
  if(NOT IS_DIRECTORY "${GUFO_NPU_PREBUILT_DESIGNS}")
    message(FATAL_ERROR "GUFO_NPU_PREBUILT_DESIGNS: no directory ${GUFO_NPU_PREBUILT_DESIGNS}")
  endif()
  set(GUFO_NPU_DESIGN_ROOT "${GUFO_NPU_PREBUILT_DESIGNS}")
  set(GUFO_NPU_DESIGNS_AVAILABLE ON)
else()
  set(GUFO_NPU_DESIGN_ROOT "${CMAKE_BINARY_DIR}/npu")
  if(GUFO_IRON_PYTHON)
    execute_process(COMMAND "${GUFO_IRON_PYTHON}" -c "import aie.iron"
      RESULT_VARIABLE _gufo_iron_missing OUTPUT_QUIET ERROR_QUIET)
    if(_gufo_iron_missing EQUAL 0)
      set(GUFO_NPU_DESIGNS_AVAILABLE ON)
    else()
      message(STATUS "NPU designs: ${GUFO_IRON_PYTHON} lacks aie.iron; skipping them")
    endif()
  endif()
endif()
if(GUFO_NPU_DESIGNS_REQUIRED AND NOT GUFO_NPU_DESIGNS_AVAILABLE)
  message(FATAL_ERROR "NPU designs required but no IRON Python found "
    "(GUFO_IRON_PYTHON='${GUFO_IRON_PYTHON}'; see BUILD-WINDOWS.md)")
endif()

set(_gufo_npu_env)
if(GUFO_NPU_DESIGNS_AVAILABLE AND NOT GUFO_NPU_PREBUILT_DESIGNS)
  if(WIN32)
    # IRON otherwise finds MinGW's c++ first on hosted Windows runners,
    # producing DLLs that require unshipped libgcc/libstdc++ runtimes.
    # IRON recognizes only the exact clang-cl/cl driver names (not
    # amdclang-cl). Its MSVC-style driver uses the static CRT for the bridge.
    find_program(GUFO_NPU_HOST_CXX NAMES clang-cl cl
      HINTS "$ENV{HIP_PATH}/lib/llvm/bin" "$ENV{HIP_PATH}/bin" REQUIRED)
  endif()
  set(GUFO_NPU_HOST_CXX "${GUFO_NPU_HOST_CXX}" CACHE FILEPATH
    "Host C++ compiler for IRON dispatch libraries")
  if(GUFO_NPU_HOST_CXX)
    list(APPEND _gufo_npu_env "CXX=${GUFO_NPU_HOST_CXX}")
  endif()
  set(_gufo_peano "$ENV{PEANO_INSTALL_DIR}")
  if(NOT _gufo_peano)
    execute_process(COMMAND "${GUFO_IRON_PYTHON}" -c
        "import sysconfig; print(sysconfig.get_paths()['purelib'])"
      OUTPUT_VARIABLE _gufo_iron_site OUTPUT_STRIP_TRAILING_WHITESPACE)
    file(TO_CMAKE_PATH "${_gufo_iron_site}/llvm-aie" _gufo_peano)
  endif()
  if(NOT IS_DIRECTORY "${_gufo_peano}")
    message(FATAL_ERROR "NPU designs: no Peano at '${_gufo_peano}'; set PEANO_INSTALL_DIR")
  endif()
  # aiecc's last step runs xclbinutil. On Windows it ships in the XRT SDK,
  # which stays off PATH (its xrt_coreutil.dll must not shadow the driver's).
  find_program(GUFO_XCLBINUTIL xclbinutil
    HINTS "${XRT_DIR}/../../.." "${XRT_DIR}/../../../bin")
  if(GUFO_XCLBINUTIL)
    get_filename_component(_gufo_xclbinutil_dir "${GUFO_XCLBINUTIL}" DIRECTORY)
    list(APPEND _gufo_npu_env "--modify" "PATH=path_list_prepend:${_gufo_xclbinutil_dir}")
  endif()
  list(APPEND _gufo_npu_env "PEANO_INSTALL_DIR=${_gufo_peano}"
    "NPU_CACHE_HOME=${GUFO_NPU_CACHE_HOME}")
endif()
if(GUFO_NPU_DESIGNS_AVAILABLE)
  add_custom_target(npu_designs)
  set_property(GLOBAL APPEND PROPERTY JOB_POOLS gufo_npu_compile=2)
endif()

# gufo_npu_design(<name> [PROGRAMS ...] [OUTPUTS ...] [EXTRA_OUTPUTS ...]
#                 [VARIANTS ...]): runs
# `<python> <kernels>/<name>/<name>.py <dir>` into ${CMAKE_BINARY_DIR}/npu/<name>,
# rerun when any file in <kernels>/<name> changes, and adds the target
# npu_design_<name> (built by default, and by npu_designs). The design writes
# <program>.xclbin and <program>_dispatch.so there for each program (default:
# just <name>), or the files OUTPUTS names, plus EXTRA_OUTPUTS. VARIANTS
# get separate build rules passing the variant as the second script argument.
# All declared files install to
# share/gufo/npu/<name> (Windows: bin/npu/<name>, beside gufo.exe). Sets
# <name>_NPU_DIR in the caller's scope.
function(gufo_npu_design name)
  if(GUFO_NPU_DESIGN AND NOT name STREQUAL GUFO_NPU_DESIGN)
    return()
  endif()
  cmake_parse_arguments(ARG "" "" "PROGRAMS;OUTPUTS;EXTRA_OUTPUTS;VARIANTS" ${ARGN})
  if(NOT ARG_PROGRAMS)
    set(ARG_PROGRAMS "${name}")
  endif()
  set(dir "${GUFO_NPU_DESIGN_ROOT}/${name}")
  set(${name}_NPU_DIR "${dir}" PARENT_SCOPE)
  if(NOT GUFO_NPU_DESIGNS_AVAILABLE)
    return()
  endif()
  # gufo::npu::FindDesign looks in <exe>/../share/gufo/npu and <exe>/npu.
  if(WIN32)
    set(destination "bin/npu/${name}")
  else()
    set(destination "share/gufo/npu/${name}")
  endif()
  set(outputs)
  foreach(program IN LISTS ARG_PROGRAMS)
    list(APPEND outputs "${program}.xclbin" "${program}_dispatch.so")
  endforeach()
  if(ARG_OUTPUTS)
    set(outputs ${ARG_OUTPUTS})
  endif()
  list(APPEND outputs ${ARG_EXTRA_OUTPUTS})
  foreach(output IN LISTS outputs)
    set_property(GLOBAL APPEND PROPERTY GUFO_NPU_ARTIFACTS "${name}/${output}")
  endforeach()
  list(TRANSFORM outputs PREPEND "${dir}/")
  if(GUFO_NPU_PREBUILT_DESIGNS)
    foreach(output IN LISTS outputs)
      if(NOT EXISTS "${output}" OR IS_DIRECTORY "${output}")
        message(FATAL_ERROR "Missing prebuilt NPU artifact: ${output}")
      endif()
      file(SIZE "${output}" size)
      if(size EQUAL 0)
        message(FATAL_ERROR "Empty prebuilt NPU artifact: ${output}")
      endif()
    endforeach()
    add_custom_target(npu_design_${name})
    add_dependencies(npu_designs npu_design_${name})
    install(FILES ${outputs} DESTINATION "${destination}")
    return()
  endif()
  set(source_dir "${GUFO_NPU_KERNELS}/${name}")
  file(GLOB_RECURSE sources CONFIGURE_DEPENDS
    "${source_dir}/*.py" "${source_dir}/*.cc" "${source_dir}/*.h")
  if(ARG_VARIANTS)
    foreach(variant IN LISTS ARG_VARIANTS)
      set(variant_outputs ${outputs})
      list(FILTER variant_outputs INCLUDE REGEX "/${name}_${variant}[._]")
      if(NOT variant_outputs)
        message(FATAL_ERROR "No declared outputs for ${name} variant ${variant}")
      endif()
      add_custom_command(
        OUTPUT ${variant_outputs}
        COMMAND ${CMAKE_COMMAND} -E env ${_gufo_npu_env}
                "${GUFO_IRON_PYTHON}" "${source_dir}/${name}.py" "${dir}" "${variant}"
        WORKING_DIRECTORY "${source_dir}"
        DEPENDS ${sources}
        JOB_POOL gufo_npu_compile
        COMMENT "Compiling NPU design ${name}_${variant}"
        VERBATIM)
    endforeach()
  elseif(NOT ARG_OUTPUTS)
    foreach(program IN LISTS ARG_PROGRAMS)
      set(program_outputs ${outputs})
      list(FILTER program_outputs INCLUDE REGEX "/${program}([.]|_dispatch[.]so$|_layout[.]txt$)")
      add_custom_command(
        OUTPUT ${program_outputs}
        COMMAND ${CMAKE_COMMAND} -E env ${_gufo_npu_env}
                "${GUFO_IRON_PYTHON}" "${source_dir}/${name}.py" "${dir}" "${program}"
        WORKING_DIRECTORY "${source_dir}"
        DEPENDS ${sources}
        JOB_POOL gufo_npu_compile
        COMMENT "Compiling NPU program ${program}"
        VERBATIM)
    endforeach()
  else()
    add_custom_command(
      OUTPUT ${outputs}
      COMMAND ${CMAKE_COMMAND} -E env ${_gufo_npu_env}
              "${GUFO_IRON_PYTHON}" "${source_dir}/${name}.py" "${dir}"
      WORKING_DIRECTORY "${source_dir}"
      DEPENDS ${sources}
      JOB_POOL gufo_npu_compile
      COMMENT "Compiling NPU design ${name}"
      VERBATIM)
  endif()
  add_custom_target(npu_design_${name} ALL DEPENDS ${outputs})
  add_dependencies(npu_designs npu_design_${name})
  install(FILES ${outputs} DESTINATION "${destination}")
endfunction()
