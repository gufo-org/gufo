# The gufo executable as the host of extension modules: modules resolve the
# public extension services from it, never from a copy of their own.
include_guard(GLOBAL)

function(gufo_extension_host target)
  # Keep every object of the libraries behind the public extension surface,
  # including ones no built-in extension uses.
  set(libraries gufo_ext_runtime gufo_logging gufo_platform gufo_core gufo_cache)
  if(ENGINE_ENABLE_HIP)
    list(APPEND libraries gufo_cache_hip gufo_hip_host)
  endif()
  set(archives)
  foreach(library IN LISTS libraries)
    list(APPEND archives "$<TARGET_FILE:${library}>")
  endforeach()
  add_dependencies(${target} ${libraries})
  if(WIN32)
    # Windows resolves a DLL's imports through the executable's import
    # library. Export Gufo's own symbols from the host archives (MSVC names
    # carry their namespace as `@gufo@@`), data with DATA, through a .def file.
    set_target_properties(${target} PROPERTIES ENABLE_EXPORTS ON)
    get_filename_component(compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    find_program(GUFO_LLVM_NM NAMES llvm-nm HINTS "${compiler_dir}" REQUIRED)
    set(definition "${CMAKE_CURRENT_BINARY_DIR}/${target}-exports.def")
    string(JOIN "|" archive_list ${archives})
    add_custom_command(OUTPUT "${definition}"
      COMMAND "${CMAKE_COMMAND}" "-DNM=${GUFO_LLVM_NM}" "-DOUTPUT=${definition}"
        "-DARCHIVES=${archive_list}"
        -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GufoWindowsExports.cmake"
      DEPENDS ${libraries} "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GufoWindowsExports.cmake"
      COMMENT "Exporting Gufo symbols for extension modules"
      VERBATIM)
    add_custom_target(${target}_exports DEPENDS "${definition}")
    add_dependencies(${target} ${target}_exports)
    target_link_options(${target} PRIVATE "LINKER:/DEF:${definition}")
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${definition}")
    foreach(archive IN LISTS archives)
      target_link_options(${target} PRIVATE "LINKER:/WHOLEARCHIVE:${archive}")
    endforeach()
    return()
  endif()
  # Export Gufo's own C++ symbols, with the typeinfo, vtables and guards that
  # keep classes and inline statics unified, but no vendored or third-party
  # code. Mangled-name globs: quoted demangled patterns match only literally.
  set(list "${CMAKE_CURRENT_BINARY_DIR}/${target}-exports.list")
  set(patterns
    _ZN4gufo* _ZNK4gufo* _ZNR4gufo* _ZNO4gufo* _ZNKR4gufo* _ZNKO4gufo*
    _ZTIN4gufo* _ZTSN4gufo* _ZTVN4gufo* _ZTTN4gufo* _ZTCN4gufo*
    _ZGVN4gufo* _ZGVZN4gufo* _ZGVZNK4gufo* _ZZN4gufo* _ZZNK4gufo*
    _ZTh*N4gufo* _ZTv*N4gufo* _ZTHN4gufo* _ZTWN4gufo*)
  list(JOIN patterns ";\n  " lines)
  file(WRITE "${list}" "{\n  ${lines};\n};\n")
  target_link_options(${target} PRIVATE "LINKER:--dynamic-list=${list}")
  set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${list}")
  target_link_options(${target} PRIVATE "LINKER:--whole-archive" ${archives}
    "LINKER:--no-whole-archive")
endfunction()
