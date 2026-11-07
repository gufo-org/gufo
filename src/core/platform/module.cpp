#include "src/core/platform/module.hpp"

#include <algorithm>
#include <cctype>
#include <string>

#include "src/core/platform/library.hpp"
#include "src/core/platform/os_info.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#else
#include <dlfcn.h>
#endif

namespace gufo::platform {
void* LoadModule(const std::filesystem::path& path) {
#ifdef _WIN32
  // The module's own directory, then the executable's: the HIP runtime and
  // the other DLLs gufo ships sit beside it.
  void* module = LoadLibraryExW(
      std::filesystem::absolute(path).c_str(), nullptr,
      LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_APPLICATION_DIR |
          LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!module)
    throw LibraryError("cannot load module " + path.string() +
                       ": Windows error " + std::to_string(GetLastError()));
#else
  // Never closed. RTLD_LOCAL keeps one module's symbols from satisfying
  // another's; the module resolves host services from the executable.
  void* module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!module)
    throw LibraryError("cannot load module " + path.string() + ": " +
                       dlerror());
#endif
  return module;
}
bool ModulesBindToExecutable(std::string_view host) {
#ifdef _WIN32
  const auto name = ExecutablePath().filename().string();
  return std::equal(name.begin(), name.end(), host.begin(), host.end(),
                    [](char a, char b) {
                      return std::tolower(static_cast<unsigned char>(a)) ==
                             std::tolower(static_cast<unsigned char>(b));
                    });
#else
  (void)host;
  return true;
#endif
}
std::string_view ModuleFileSuffix() noexcept {
#ifdef _WIN32
  return ".dll";
#else
  return ".so";
#endif
}
void* ModuleSymbol(void* module, const char* name) {
#ifdef _WIN32
  return reinterpret_cast<void*>(
      GetProcAddress(static_cast<HMODULE>(module), name));
#else
  return dlsym(module, name);
#endif
}
}  // namespace gufo::platform
