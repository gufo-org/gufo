#pragma once

#include <string_view>

#include "src/core/logging/logging.hpp"

namespace gufo::ext {
using LogLevel = server::LogLevel;

// A borrowed host context. The caller owns context and must outlive its use.
// No HTTP, scheduler or model dependency is part of this initial handle.
struct Host {
  using LogCallback = void (*)(void*, LogLevel, std::string_view,
                               std::string_view);
  void* context = nullptr;
  LogCallback log = nullptr;
  void Log(LogLevel level, std::string_view component,
           std::string_view message) const;
};
}  // namespace gufo::ext
