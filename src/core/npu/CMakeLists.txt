# The NPU runtime on XRT (found at the top level). Only the backend sources
# include XRT; npu.hpp names no XRT type.
# Which NPUs are present: sysfs on Linux, XRT's enumeration on Windows.
add_library(gufo_npu STATIC common.cpp xrt.cpp q8_gemm.cpp
  "$<$<NOT:$<PLATFORM_ID:Windows>>:${CMAKE_CURRENT_SOURCE_DIR}/probe.cpp>"
  "$<$<PLATFORM_ID:Windows>:${CMAKE_CURRENT_SOURCE_DIR}/probe_windows.cpp>")
target_include_directories(gufo_npu PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_npu PUBLIC cxx_std_20)
target_link_libraries(gufo_npu PUBLIC gufo_platform PRIVATE XRT::xrt_coreutil)
# Windows: use the xrt_coreutil.dll the NPU driver installs in System32; a
# copy beside gufo.exe cannot find the driver's device shim. Delay-loaded, so
# Gufo starts without the driver (LoadXrt, xrt_load.hpp).
if(WIN32)
  target_link_options(gufo_npu INTERFACE "/DELAYLOAD:xrt_coreutil.dll")
  target_link_libraries(gufo_npu PUBLIC delayimp)
endif()
