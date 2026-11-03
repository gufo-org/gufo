# Evaluate in each caller directory: CMake variables have directory scope.

# Source archives and Nix builds have the same API identity as Git checkouts.
# Records are permanent; adding one advances the revision without a shared edit.
set(_gufo_api_root "${CMAKE_CURRENT_LIST_DIR}/../include/gufo/ext")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${_gufo_api_root}/api-series.json")
file(READ "${_gufo_api_root}/api-series.json" _gufo_api_series)
string(JSON _gufo_api_fields LENGTH "${_gufo_api_series}")
if(NOT _gufo_api_fields EQUAL 3)
  message(FATAL_ERROR "extension API: api-series.json requires exactly major, minor, revision_base")
endif()
foreach(field major minor revision_base)
  string(JSON value ERROR_VARIABLE error GET "${_gufo_api_series}" "${field}")
  string(JSON type ERROR_VARIABLE type_error TYPE "${_gufo_api_series}" "${field}")
  if(error OR type_error OR NOT type STREQUAL "NUMBER" OR NOT value MATCHES "^(0|[1-9][0-9]*)$")
    message(FATAL_ERROR "extension API: invalid ${field} in api-series.json")
  endif()
  set(GUFO_EXTENSION_API_${field} "${value}")
endforeach()
file(GLOB _gufo_api_records CONFIGURE_DEPENDS
  "${_gufo_api_root}/changes/${GUFO_EXTENSION_API_major}.${GUFO_EXTENSION_API_minor}/*.md")
# The baseline record documents that revision; subsequent records advance it.
set(_gufo_api_revisions ${_gufo_api_records})
list(FILTER _gufo_api_revisions EXCLUDE REGEX "/baseline[.]md$")
# Legacy source fixtures may still use a numbered baseline record.
list(FILTER _gufo_api_revisions EXCLUDE REGEX "/${GUFO_EXTENSION_API_revision_base}[.]md$")
list(LENGTH _gufo_api_revisions _gufo_api_count)
foreach(record IN LISTS _gufo_api_records)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${record}")
  file(READ "${record}" content)
  string(STRIP "${content}" content)
  if(content STREQUAL "")
    message(FATAL_ERROR "extension API: empty change record ${record}")
  endif()
endforeach()
math(EXPR GUFO_EXTENSION_API_revision "${GUFO_EXTENSION_API_revision_base}+${_gufo_api_count}")
set(GUFO_EXTENSION_API_VERSION
  "${GUFO_EXTENSION_API_major}.${GUFO_EXTENSION_API_minor}.${GUFO_EXTENSION_API_revision}")
