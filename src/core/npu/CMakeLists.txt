# The NPU runtime on XRT (found at the top level). Only xrt_backend.cpp
# includes XRT; npu.hpp names no XRT type, and xrt.cpp reaches XRT through
# the backend's C interface (xrt_backend.hpp).
# Which NPUs are present: sysfs on Linux, XRT's enumeration on Windows.
add_library(gufo_npu STATIC common.cpp xrt.cpp q8_gemm.cpp bfp_gemm.cpp host_ops.cpp qwen3_decoder.cpp
  qwen3_prefill.cpp
  "$<$<NOT:$<PLATFORM_ID:Windows>>:${CMAKE_CURRENT_SOURCE_DIR}/probe.cpp>"
  "$<$<PLATFORM_ID:Windows>:${CMAKE_CURRENT_SOURCE_DIR}/probe_windows.cpp>")
target_include_directories(gufo_npu PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_npu PUBLIC cxx_std_20)
target_link_libraries(gufo_npu PUBLIC gufo_platform)
if(WIN32)
  # Use the xrt_coreutil.dll the NPU driver installs in System32; a copy
  # beside gufo.exe cannot find the driver's device shim. Delay-loaded, so
  # Gufo starts without the driver (xrt_load.hpp).
  target_sources(gufo_npu PRIVATE xrt_backend.cpp)
  target_link_libraries(gufo_npu PRIVATE XRT::xrt_coreutil)
  target_link_options(gufo_npu INTERFACE "/DELAYLOAD:xrt_coreutil.dll")
  target_link_libraries(gufo_npu PUBLIC delayimp)
else()
  # A module loaded on first NPU use (probe.cpp), so Gufo neither needs XRT
  # to start nor searches XRT's library directory for its other libraries.
  # It exports only its C entry point: Gufo exports its own C++ symbols to
  # extension modules, and none of this module's code may bind to them.
  add_library(gufo_npu_xrt MODULE xrt_backend.cpp)
  target_include_directories(gufo_npu_xrt PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_features(gufo_npu_xrt PRIVATE cxx_std_20)
  target_link_libraries(gufo_npu_xrt PRIVATE XRT::xrt_coreutil)
  target_link_options(gufo_npu_xrt PRIVATE "LINKER:--no-undefined")
  set_target_properties(gufo_npu_xrt PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
  install(TARGETS gufo_npu_xrt LIBRARY DESTINATION lib/gufo)
  add_dependencies(gufo_npu gufo_npu_xrt)
  target_compile_definitions(gufo_npu PRIVATE
    "GUFO_NPU_XRT_BUILD_MODULE=\"$<TARGET_FILE:gufo_npu_xrt>\"")
endif()
# Host operators: AVX2 + FMA (every Strix Halo and development x86 CPU).
if(MSVC)
  set_source_files_properties(host_ops.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  set_source_files_properties(host_ops.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-O3")
endif()
target_compile_definitions(gufo_npu PRIVATE "GUFO_NPU_BUILD_DESIGNS=\"${GUFO_NPU_DESIGN_ROOT}\"")
