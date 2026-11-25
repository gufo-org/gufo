# The gufo executable as the host of extension modules: modules resolve the
# public extension services from it, never from a copy of their own.
include_guard(GLOBAL)

# Gufo's static libraries behind the module-linkable exports: the Qwen family
# library and the enabled gufo:: exports of the in-tree extensions built into
# the executable, with the gufo_* libraries they link, transitively, up to the
# libraries in `stop` (already kept whole). Two assumptions make the code a
# module reaches through those headers resolve from the executable:
#  - every library in that closure is a gufo_* or manifest target, kept whole
#    here; any other non-imported static or object library fails configure,
#    since its objects could be missing or unexported;
#  - the code is in the gufo namespace, which the Linux dynamic list exports.
#    Exported headers are the SURFACE closure, which the extension API check
#    audits; extern "C" or global-namespace symbols there are not exported.
function(_gufo_host_export_libraries out)
  set(stop ${ARGN})
  set(queue)
  if(TARGET gufo_qwen_family)
    list(APPEND queue "gufo_qwen_family|gufo::qwen_family")
  endif()
  _gufo_module_host_extensions(hosts)
  foreach(name IN LISTS hosts)
    _gufo_enabled_exports(exports "${name}")
    foreach(export IN LISTS exports)
      get_property(exported GLOBAL PROPERTY "GUFO_EXT_EXPORT_${export}_TARGET")
      list(APPEND queue "${exported}|gufo::${export}")
    endforeach()
  endforeach()
  set(visited)
  set(libraries)
  while(queue)
    list(POP_FRONT queue entry)
    string(REGEX MATCH "^([^|]+)\\|(.*)$" ignored "${entry}")
    set(current "${CMAKE_MATCH_1}")
    set(root "${CMAKE_MATCH_2}")
    if(NOT TARGET "${current}")
      continue()
    endif()
    get_target_property(aliased "${current}" ALIASED_TARGET)
    if(aliased)
      set(current "${aliased}")
    endif()
    get_target_property(imported "${current}" IMPORTED)
    if(imported OR current IN_LIST visited OR current IN_LIST stop)
      continue()
    endif()
    list(APPEND visited "${current}")
    get_target_property(type "${current}" TYPE)
    get_property(manifest_target GLOBAL PROPERTY "GUFO_EXT_TARGET_${current}_OWNER" SET)
    if(NOT current MATCHES "^gufo_" AND NOT manifest_target)
      if(type MATCHES "^(STATIC_LIBRARY|OBJECT_LIBRARY)$")
        message(FATAL_ERROR "${root} links ${current}, a ${type} outside Gufo's gufo_* and manifest libraries; "
          "the gufo executable cannot keep and export its code for loadable modules")
      endif()
      continue()
    endif()
    set(items)
    if(type STREQUAL STATIC_LIBRARY)
      list(APPEND libraries "${current}")
      get_target_property(values "${current}" LINK_LIBRARIES)
      if(values)
        list(APPEND items ${values})
      endif()
    endif()
    get_target_property(values "${current}" INTERFACE_LINK_LIBRARIES)
    if(values)
      list(APPEND items ${values})
    endif()
    _gufo_link_names(items ${items})
    foreach(item IN LISTS items)
      list(APPEND queue "${item}|${root}")
    endforeach()
  endwhile()
  set(${out} "${libraries}" PARENT_SCOPE)
endfunction()

function(gufo_extension_host target)
  # Keep every object of the libraries behind the public extension surface,
  # including ones no built-in extension uses.
  set(libraries gufo_ext_runtime gufo_logging gufo_json gufo_safetensors gufo_platform
    gufo_npu gufo_core gufo_cache)
  if(ENGINE_ENABLE_HIP)
    list(APPEND libraries gufo_cache_hip gufo_hip_host)
  endif()
  # Modules link the Qwen family and built-in extensions' exports compile-only
  # (gufo_extension_target), so their objects stay here too. Not on Windows
  # yet: modules there cannot link them.
  if(NOT WIN32)
    _gufo_host_export_libraries(exported ${libraries})
    if(exported)
      # Linked normally as well, so their own dependencies are on the link line.
      target_link_libraries(${target} PRIVATE ${exported})
      list(APPEND libraries ${exported})
    endif()
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
