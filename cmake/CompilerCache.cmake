# Shared by the application and standalone contract fixtures.
find_program(GUFO_CCACHE_PROGRAM NAMES ccache)
if (GUFO_CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${GUFO_CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${GUFO_CCACHE_PROGRAM}")
  # Newer CMake gives clang-cl gcc-style depfiles (-clang:-MD), which ccache
  # rejects as unsupported; /showIncludes is cacheable.
  if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(GUFO_CCACHE_CLANG_CL ON)
    foreach(lang C CXX)
      set(CMAKE_DEPFILE_FLAGS_${lang} "/showIncludes")
      set(CMAKE_${lang}_DEPFILE_FORMAT msvc)
    endforeach()
  endif()
  message(STATUS "gufo: compiling C and CXX through ${GUFO_CCACHE_PROGRAM}")
else()
  message(STATUS "gufo: ccache not found, compiling without a launcher")
endif()
