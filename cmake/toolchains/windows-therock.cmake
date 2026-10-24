# Windows host + TheRock HIP SDK toolchain, chain-loaded by vcpkg from the
# windows-base preset. HIP_PATH names the SDK root; nothing is hard-coded.
if (NOT DEFINED ENV{HIP_PATH})
  message(FATAL_ERROR "Set HIP_PATH to the TheRock/HIP SDK root (see BUILD-WINDOWS.md)")
endif()
file(TO_CMAKE_PATH "$ENV{HIP_PATH}" GUFO_HIP_ROOT)
# TheRock keeps clang under lib/llvm/bin (some builds ship it only as
# amdclang-cl.exe); the HIP SDK installer under bin.
foreach(candidate "${GUFO_HIP_ROOT}/lib/llvm/bin/clang-cl.exe"
                  "${GUFO_HIP_ROOT}/lib/llvm/bin/amdclang-cl.exe"
                  "${GUFO_HIP_ROOT}/bin/clang-cl.exe")
  if (EXISTS "${candidate}")
    set(GUFO_CLANG_CL "${candidate}")
    break()
  endif()
endforeach()
if (NOT GUFO_CLANG_CL)
  message(FATAL_ERROR "No clang-cl.exe under HIP_PATH=${GUFO_HIP_ROOT}")
endif()
set(CMAKE_C_COMPILER "${GUFO_CLANG_CL}")
set(CMAKE_CXX_COMPILER "${GUFO_CLANG_CL}")
set(CMAKE_HIP_COMPILER "${GUFO_CLANG_CL}")
list(APPEND CMAKE_PREFIX_PATH "${GUFO_HIP_ROOT}")
set(ROCM_PATH "${GUFO_HIP_ROOT}" CACHE PATH "HIP runtime root")
