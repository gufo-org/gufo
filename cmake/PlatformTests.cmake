# Tests of the OS leaves, shared by the Linux build and the Windows
# platform-only build (GUFO_PLATFORM_ONLY).
add_executable(platform_mapped_file_test
  tests/core/platform_mapped_file_test.cpp src/core/platform/mapped_file.cpp)
add_test(NAME platform_mapped_file_test COMMAND platform_mapped_file_test)
set_tests_properties(platform_mapped_file_test PROPERTIES
  LABELS "cpu;platform")
add_executable(platform_fs_test
  tests/core/platform_fs_test.cpp
  "$<$<NOT:$<PLATFORM_ID:Windows>>:src/core/platform/fs_posix.cpp>"
  "$<$<PLATFORM_ID:Windows>:src/core/platform/fs_windows.cpp>"
  src/core/platform/mapped_file.cpp)
add_test(NAME platform_fs_test COMMAND platform_fs_test)
set_tests_properties(platform_fs_test PROPERTIES LABELS "cpu;platform")
