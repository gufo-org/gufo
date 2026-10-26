#ifndef GUFO_CORE_PLATFORM_ENV_HPP_
#define GUFO_CORE_PLATFORM_ENV_HPP_

#include <string_view>

namespace gufo::platform {

/// Sets an environment variable in this process, overwriting any previous
/// value. Returns false when the OS rejects the name or value. An empty
/// value removes the variable on Windows (the CRT cannot store one);
/// POSIX stores it empty.
bool SetEnv(std::string_view name, std::string_view value);

/// Removes an environment variable from this process. Removing an absent
/// variable succeeds. Returns false when the OS rejects the name.
bool UnsetEnv(std::string_view name);

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_ENV_HPP_
