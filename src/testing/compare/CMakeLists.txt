# Shared logit comparison support: the reference/candidate tolerance reporter
# used by `gufo bench` and the model oracle tests.
#
# Card 08 (RFC step 5) lifted it out of gufo_core so a comparison helper is not
# part of the shipped engine library's object set. `library.json` records which
# targets own it. It is built in production and CPU builds alike because
# `src/cli/bench/bench.cpp` calls CompareLogits.
add_library(gufo_compare STATIC logit_comparator.cpp)

target_include_directories(gufo_compare PUBLIC ${CMAKE_SOURCE_DIR})
# The sources need no core symbol today; the edge keeps the compile environment
# (standard, defines, OpenMP) these translation units had inside gufo_core.
target_link_libraries(gufo_compare PUBLIC gufo_core)
