add_library(gufo_cache STATIC completion.cpp identity.cpp)
target_include_directories(gufo_cache PUBLIC ${PROJECT_SOURCE_DIR})
