#pragma once

#include <span>

namespace gufo::cli {
// `gufo extension install|list|update|remove` (gufo-org/gufo#1075).
int RunExtension(std::span<const char* const> args);
}  // namespace gufo::cli
