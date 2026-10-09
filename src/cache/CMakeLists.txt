add_library(gufo_cache STATIC completion.cpp identity.cpp ledger.cpp)
target_include_directories(gufo_cache PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_cache PUBLIC Threads::Threads)
