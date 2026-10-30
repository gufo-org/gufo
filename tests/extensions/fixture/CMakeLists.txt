cmake_minimum_required(VERSION 3.21)
project(extension_fixture LANGUAGES CXX)
get_filename_component(GUFO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
# Reuse the production targets; they intentionally resolve their source root
# through PROJECT_SOURCE_DIR, as in the root build.
set(PROJECT_SOURCE_DIR "${GUFO_ROOT}")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)
include(CTest)
add_subdirectory("${GUFO_ROOT}/src/core/platform" platform)
add_subdirectory("${GUFO_ROOT}/src/ext" runtime)
add_subdirectory("${GUFO_ROOT}/tests/extensions" extensions)
