#pragma once

#include <string>

namespace gufo::cli {
// These inspect the generated registry without loading models or probing HIP.
std::string ExtensionDiagnostics(bool json);
}  // namespace gufo::cli
