# RFC step 6: nothing below the extension runtime may reach the serving or HTTP
# tiers. CMake knows the real dependency edges, so it writes them out and the
# `extension_link_graph_gate` test fails on a forbidden one.
#
# The report lists each protected target with its transitive link closure, so
# the check also catches an edge that only appears through an intermediate
# library. Coverage is limited to fixed names and manifest-declared libraries.

set(GUFO_LINK_GRAPH_PROTECTED
  gufo_platform
  gufo_logging
  gufo_hip_host
  gufo_cache
  gufo_cache_hip
  gufo_core
  gufo_ext_runtime)
set(GUFO_LINK_GRAPH_FORBIDDEN gufo_serving gufo_http)

# Link items that name a system library rather than a project target. The
# platform leaf's own list joins it, so a new OS library stays classified.
set(GUFO_LINK_GRAPH_SYSTEM_LIBS
  dl m pthread rt atomic stdc++ gcc_s
  ws2_32 iphlpapi bcrypt advapi32)
if (DEFINED GUFO_PLATFORM_LIBS)
  list(APPEND GUFO_LINK_GRAPH_SYSTEM_LIBS ${GUFO_PLATFORM_LIBS})
endif()
list(REMOVE_DUPLICATES GUFO_LINK_GRAPH_SYSTEM_LIBS)

# One INTERFACE_LINK_LIBRARIES entry per line, with the generated-expression
# wrappers CMake uses for static private edges stripped off.
function(gufo_link_graph_entry TARGET OUT)
  get_target_property(interface ${TARGET} INTERFACE_LINK_LIBRARIES)
  if(interface STREQUAL "interface-NOTFOUND" OR interface STREQUAL "")
    set(${OUT} "" PARENT_SCOPE)
    return()
  endif()
  set(entries "")
  foreach(item ${interface})
    # Only plain target names and the LINK_ONLY/BUILD_INTERFACE wrappers CMake
    # emits for static libraries count. Known system libraries, absolute paths
    # and linker flags are classified separately; everything else fails.
    if(item MATCHES "^::@(\\([^)]*\\))?$")
      # CMake's own item-group delimiters: a directory that links into a target
      # defined elsewhere gets its entries wrapped in a group opened by
      # `::@(<id>)` and closed by `::@`. The markers carry no dependency, while
      # the target names inside them stay ordinary list items, so skipping the
      # markers hides no edge. Verified on this tree: the groups on
      # gufo_qwen3_tts, gufo_qwen38_flash_next, gufo_deepseek_v4_flash and
      # gufo_qwen_image_21 still list gufo_core, gufo_logging,
      # gufo_qwen_kernels, gufo_qwen_control_tokens and gufo_qwen_text
      # individually, and the generated report keeps each of them.
      continue()
    endif()
    if(TARGET "${item}")
      list(APPEND entries "${item}")
    elseif("${item}" MATCHES "^\\$<(LINK_ONLY|BUILD_INTERFACE):([^>]+)>$")
      # CMAKE_MATCH_ is only valid after the enclosing if() finished, so the
      # target test goes in its own command.
      if(TARGET "${CMAKE_MATCH_2}")
        list(APPEND entries "${CMAKE_MATCH_2}")
      else()
        message(FATAL_ERROR "Link graph: unclassified interface item '${item}' on ${TARGET}")
      endif()
    elseif(IS_ABSOLUTE "${item}" OR "${item}" MATCHES "^-" OR
           "${item}" IN_LIST GUFO_LINK_GRAPH_SYSTEM_LIBS)
      # Classified linker flags, absolute library paths and system libraries
      # carry no project-target dependency. Generator expressions never qualify.
      if("${item}" MATCHES "\\$<")
        message(FATAL_ERROR "Link graph: unclassified interface item '${item}' on ${TARGET}")
      endif()
    else()
      message(FATAL_ERROR "Link graph: unclassified interface item '${item}' on ${TARGET}")
    endif()
  endforeach()
  set(${OUT} "${entries}" PARENT_SCOPE)
endfunction()

function(gufo_link_graph_closure TARGET OUT)
  set(frontier "${TARGET}")
  set(seen "")
  while(frontier)
    list(GET frontier 0 current)
    list(REMOVE_AT frontier 0)
    if(current IN_LIST seen)
      continue()
    endif()
    list(APPEND seen "${current}")
    gufo_link_graph_entry("${current}" direct)
    foreach(next ${direct})
      if(NOT next IN_LIST seen)
        list(APPEND frontier "${next}")
      endif()
    endforeach()
  endwhile()
  list(REMOVE_ITEM seen "${TARGET}")
  list(SORT seen)
  set(${OUT} "${seen}" PARENT_SCOPE)
endfunction()

# Writes ${OUTPUT} with every protected target and its closure. Model and family
# libraries join the protected set from their own manifests, so a new library is
# covered without editing this file.
function(gufo_write_link_graph OUTPUT)
  set(protected ${GUFO_LINK_GRAPH_PROTECTED})
  # Same rule as tools/ci/check-target-graph.py: every declared library joins
  # the protected set, wherever its manifest lives under src/.
  file(GLOB_RECURSE manifests CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*/library.json")
  foreach(manifest ${manifests})
    file(READ "${manifest}" text)
    if(text MATCHES "[\n]  \"target\": \"([A-Za-z0-9_]+)\"")
      list(APPEND protected "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  list(SORT protected)

  # A library can be legitimately undefined in this configuration (the HIP
  # tiers in a HIP-off build), so it is reported as absent instead of silently
  # dropped. Anything neither built nor recorded fails the gate.
  set(absent "")
  if(ENGINE_ENABLE_HIP)
    set(hip_json true)
  else()
    set(hip_json false)
  endif()
  set(json "{\n  \"hip\": ${hip_json},\n  \"forbidden\": [")
  list(JOIN GUFO_LINK_GRAPH_FORBIDDEN "\", \"" forbidden)
  string(APPEND json "\"${forbidden}\"],\n  \"targets\": {")
  set(first TRUE)
  foreach(target ${protected})
    if(NOT TARGET ${target})
      list(APPEND absent "${target}")
      continue()
    endif()
    gufo_link_graph_closure("${target}" closure)
    list(JOIN closure "\", \"" joined)
    if(joined STREQUAL "")
      set(entry "")
    else()
      set(entry "\"${joined}\"")
    endif()
    if(first)
      set(first FALSE)
    else()
      string(APPEND json ",")
    endif()
    string(APPEND json "\n    \"${target}\": [${entry}]")
  endforeach()
  string(APPEND json "},\n  \"absent\": [")
  list(SORT absent)
  list(JOIN absent "\", \"" joined_absent)
  if(joined_absent STREQUAL "")
    string(APPEND json "]\n}\n")
  else()
    string(APPEND json "\"${joined_absent}\"]\n}\n")
  endif()
  file(WRITE "${OUTPUT}" "${json}")
endfunction()
