#ifndef GUFO_CORE_PLATFORM_MODULE_HPP_
#define GUFO_CORE_PLATFORM_MODULE_HPP_

#include <filesystem>
#include <string_view>

namespace gufo::platform {
/// Loads a module that stays mapped for the rest of the process: its code may
/// back providers, sessions and device kernels long after registration.
/// Unresolved symbols fail the load. Throws LibraryError.
[[nodiscard]] void* LoadModule(const std::filesystem::path& path);
/// Returns the exported symbol, or nullptr when the module does not define it.
[[nodiscard]] void* ModuleSymbol(void* module, const char* name);
/// Whether modules built against the executable file `host` bind to this
/// process. Windows modules import from that file by name, so a renamed copy
/// of it would map a second image; elsewhere they bind to any executable.
[[nodiscard]] bool ModulesBindToExecutable(std::string_view host);
/// File suffix of a loadable module on this platform: ".so" or ".dll".
[[nodiscard]] std::string_view ModuleFileSuffix() noexcept;
}  // namespace gufo::platform
#endif
