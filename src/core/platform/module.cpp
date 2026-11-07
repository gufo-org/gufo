#include "src/core/platform/module.hpp"

#include <string>

#include "src/core/platform/library.hpp"

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
  void* module = LoadLibraryExW(
      path.c_str(), nullptr,
      LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
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
void* ModuleSymbol(void* module, const char* name) {
#ifdef _WIN32
  return reinterpret_cast<void*>(
      GetProcAddress(static_cast<HMODULE>(module), name));
#else
  return dlsym(module, name);
#endif
}
}  // namespace gufo::platform
