#pragma once

#include <string_view>

namespace gufo::ext {
// Pre-1.0 compatibility is one minor series and an additive revision floor.
inline constexpr unsigned kApiMajor = 0;
inline constexpr unsigned kApiMinor = 1;
inline constexpr unsigned kApiRevision = 6;
inline constexpr std::string_view kApiVersion = "0.1.6";
std::string_view RuntimeApiVersion();
}  // namespace gufo::ext
