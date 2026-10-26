#include "src/core/platform/env.hpp"

#include <string>

#if defined(_WIN32)
#include <stdlib.h>
#else
#include <cstdlib>
#endif

namespace gufo::platform {

bool SetEnv(std::string_view name, std::string_view value) {
#if defined(_WIN32)
  return ::_putenv_s(std::string(name).c_str(), std::string(value).c_str()) ==
         0;
#else
  return ::setenv(std::string(name).c_str(), std::string(value).c_str(), 1) ==
         0;
#endif
}

bool UnsetEnv(std::string_view name) {
#if defined(_WIN32)
  return ::_putenv(std::string(name).append("=").c_str()) == 0;
#else
  return ::unsetenv(std::string(name).c_str()) == 0;
#endif
}

}  // namespace gufo::platform
