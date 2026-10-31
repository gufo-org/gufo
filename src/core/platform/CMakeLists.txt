# The OS leaves. The Linux build and the Windows platform-only build
# (GUFO_PLATFORM_ONLY) share this list.
add_library(gufo_platform STATIC
  env.cpp
  library.cpp
  mapped_file.cpp
  net.cpp
  shutdown_signal.cpp
  publish.cpp
  os_info.cpp
  "$<$<NOT:$<PLATFORM_ID:Windows>>:${CMAKE_CURRENT_SOURCE_DIR}/fs_posix.cpp>"
  "$<$<NOT:$<PLATFORM_ID:Windows>>:${CMAKE_CURRENT_SOURCE_DIR}/pages_posix.cpp>"
  "$<$<NOT:$<PLATFORM_ID:Windows>>:${CMAKE_CURRENT_SOURCE_DIR}/page_pin_linux.cpp>"
  "$<$<PLATFORM_ID:Windows>:${CMAKE_CURRENT_SOURCE_DIR}/fs_windows.cpp>"
  "$<$<PLATFORM_ID:Windows>:${CMAKE_CURRENT_SOURCE_DIR}/pages_windows.cpp>"
  "$<$<PLATFORM_ID:Windows>:${CMAKE_CURRENT_SOURCE_DIR}/page_pin_windows.cpp>")
target_include_directories(gufo_platform PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_platform PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
if (WIN32)
  # WinSock (net), process memory counters (logging), BCrypt and the registry
  # (os_info). The leaf owns them so every project adding it links them,
  # including the platform-only build and the isolated extension fixture.
  set(GUFO_PLATFORM_LIBS ws2_32 iphlpapi bcrypt advapi32)
  set(GUFO_PLATFORM_LIBS ${GUFO_PLATFORM_LIBS} PARENT_SCOPE)
  target_link_libraries(gufo_platform PUBLIC ${GUFO_PLATFORM_LIBS})
endif()
