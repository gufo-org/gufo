#ifndef GUFO_CORE_PLATFORM_MODULE_HPP_
#define GUFO_CORE_PLATFORM_MODULE_HPP_

#include <filesystem>

namespace gufo::platform {
/// Loads a module that stays mapped for the rest of the process: its code may
/// back providers, sessions and device kernels long after registration.
/// Unresolved symbols fail the load. Throws LibraryError.
[[nodiscard]] void* LoadModule(const std::filesystem::path& path);
/// Returns the exported symbol, or nullptr when the module does not define it.
[[nodiscard]] void* ModuleSymbol(void* module, const char* name);
}  // namespace gufo::platform
#endif
