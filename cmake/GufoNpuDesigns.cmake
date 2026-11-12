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

# gufo_npu_design(<name> SCRIPT <design.py> [SOURCES <files>...]): runs
# `<python> <design.py> <dir>` into ${CMAKE_BINARY_DIR}/npu/<name> and adds the
# target npu_design_<name>. The design writes <name>.xclbin and its dispatch
# library there. Sets <name>_NPU_DIR in the caller's scope.
function(gufo_npu_design name)
  cmake_parse_arguments(ARG "" "SCRIPT" "SOURCES" ${ARGN})
  set(dir "${CMAKE_BINARY_DIR}/npu/${name}")
  set(${name}_NPU_DIR "${dir}" PARENT_SCOPE)
  if(NOT GUFO_NPU_DESIGNS_AVAILABLE)
    return()
  endif()
  get_filename_component(script "${ARG_SCRIPT}" ABSOLUTE)
  get_filename_component(script_dir "${script}" DIRECTORY)
  add_custom_command(
    OUTPUT "${dir}/${name}.xclbin"
    COMMAND ${CMAKE_COMMAND} -E env "PEANO_INSTALL_DIR=$ENV{PEANO_INSTALL_DIR}"
            "${GUFO_IRON_PYTHON}" "${script}" "${dir}"
    WORKING_DIRECTORY "${script_dir}"
    DEPENDS "${script}" ${ARG_SOURCES}
    COMMENT "Compiling NPU design ${name}"
    VERBATIM)
  add_custom_target(npu_design_${name} DEPENDS "${dir}/${name}.xclbin")
endfunction()
