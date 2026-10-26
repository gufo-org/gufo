#pragma once

// HIP's host headers define __noinline__ as a macro for non-HIP C++ sources.
// libstdc++ 16 declares [[__gnu__::__noinline__]] in <format>, which that
// macro breaks under both Clang and GCC. CMake force-includes this header in
// every Linux C++ source so <format> is parsed before any HIP header.
#if __has_include(<format>)
#include <format>
#endif
