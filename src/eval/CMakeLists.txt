# Evaluation dataset, answer extraction and regrading.
#
# Card 08 (RFC step 5) extracted these from gufo_core into the eval target the
# `gufo eval` command and the eval tests link. The HTTP client stays compiled
# into the `gufo` executable because eval remains a core HTTP client; the
# default dataset is a core fixture (`tests/fixtures/eval/antirez-ds4.json`),
# so `gufo eval` keeps working in builds without DeepSeek.
add_library(gufo_eval STATIC
  dataset.cpp
  extract.cpp
  regrade.cpp)

target_include_directories(gufo_eval PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(gufo_eval PUBLIC gufo_core)
