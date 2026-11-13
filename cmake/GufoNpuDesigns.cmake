# NPU designs (IRON, src/core/npu/kernels) compiled at build time when an
# mlir-aie Python is available: GUFO_IRON_PYTHON (the dev shell sets it), with
# Peano from PEANO_INSTALL_DIR. Without one, designs and their device tests
# are skipped; runtimes then load designs from directories given at run time.
include_guard(GLOBAL)

set(GUFO_IRON_PYTHON "$ENV{GUFO_IRON_PYTHON}" CACHE FILEPATH
  "Python with mlir-aie (IRON) for compiling NPU designs; empty skips them")
set(GUFO_NPU_DESIGNS_AVAILABLE OFF)
if(GUFO_IRON_PYTHON)
  execute_process(COMMAND "${GUFO_IRON_PYTHON}" -c "import aie.iron"
    RESULT_VARIABLE _gufo_iron_missing OUTPUT_QUIET ERROR_QUIET)
  if(_gufo_iron_missing EQUAL 0)
    set(GUFO_NPU_DESIGNS_AVAILABLE ON)
  else()
    message(STATUS "NPU designs: ${GUFO_IRON_PYTHON} lacks aie.iron; skipping them")
  endif()
endif()

# gufo_npu_design(<name> SCRIPT <design.py> [SOURCES <files>...]
#                 [PROGRAMS <names>...]): runs `<python> <design.py> <dir>` into
# ${CMAKE_BINARY_DIR}/npu/<name> and adds the target npu_design_<name> (built
# by default). The design writes <program>.xclbin and <program>_dispatch.so
# there for each program (default: just <name>); all install to
# share/gufo/npu/<name>. Sets <name>_NPU_DIR in the caller's scope.
function(gufo_npu_design name)
  cmake_parse_arguments(ARG "" "SCRIPT" "SOURCES;PROGRAMS" ${ARGN})
  if(NOT ARG_PROGRAMS)
    set(ARG_PROGRAMS "${name}")
  endif()
  set(dir "${CMAKE_BINARY_DIR}/npu/${name}")
  set(${name}_NPU_DIR "${dir}" PARENT_SCOPE)
  if(NOT GUFO_NPU_DESIGNS_AVAILABLE)
    return()
  endif()
  get_filename_component(script "${ARG_SCRIPT}" ABSOLUTE)
  get_filename_component(script_dir "${script}" DIRECTORY)
  # IRON caches compiled kernel objects under NPU_CACHE_HOME; keep them in
  # the build tree (sandboxed builds have no writable home).
  set(outputs)
  foreach(program IN LISTS ARG_PROGRAMS)
    list(APPEND outputs "${dir}/${program}.xclbin" "${dir}/${program}_dispatch.so")
  endforeach()
  add_custom_command(
    OUTPUT ${outputs}
    COMMAND ${CMAKE_COMMAND} -E env "PEANO_INSTALL_DIR=$ENV{PEANO_INSTALL_DIR}"
            "NPU_CACHE_HOME=${CMAKE_BINARY_DIR}/npu/.cache"
            "${GUFO_IRON_PYTHON}" "${script}" "${dir}"
    WORKING_DIRECTORY "${script_dir}"
    DEPENDS "${script}" ${ARG_SOURCES}
    COMMENT "Compiling NPU design ${name}"
    VERBATIM)
  add_custom_target(npu_design_${name} ALL DEPENDS ${outputs})
  if(NOT WIN32)
    # Each program's xclbin and dispatch library, and any host-side layout
    # files (<name>_*.txt).
    install(FILES ${outputs} DESTINATION "share/gufo/npu/${name}")
    install(DIRECTORY "${dir}/" DESTINATION "share/gufo/npu/${name}"
      FILES_MATCHING PATTERN "${name}_*.txt" PATTERN "*.prj" EXCLUDE)
  endif()
endfunction()
