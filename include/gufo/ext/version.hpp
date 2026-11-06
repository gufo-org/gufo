#pragma once

#include <gufo/ext/detail/api_version.hpp>

namespace gufo::ext {
// Pre-1.0 compatibility is one minor series and an additive revision floor.
std::string_view RuntimeApiVersion();
}  // namespace gufo::ext
