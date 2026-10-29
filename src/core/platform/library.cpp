#include "src/core/platform/library.hpp"

#include <filesystem>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#else
#include <dlfcn.h>
#endif

namespace gufo::platform {
Library::Library(const std::string& name) {
#ifdef _WIN32
  // Prefer the packaged DLL beside the executable. Development/test builds
  // also have a configured absolute library path; neither mode searches PATH
  // or the current working directory for a runtime dependency.
  auto path = std::filesystem::path(name);
  path.make_preferred();
  handle_ = LoadLibraryExW(
      path.filename().c_str(), nullptr,
      LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!handle_ && path.is_absolute())
    handle_ = LoadLibraryExW(
        path.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!handle_)
    throw LibraryError("cannot load runtime library " + name +
                       ": Windows error " + std::to_string(GetLastError()));
#else
  handle_ = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
  // Portable packages relocate the pinned closure and supply its library
  // directories through the ELF loader. The build-host Nix path is absent.
  if (!handle_ && std::filesystem::path(name).is_absolute())
    handle_ = dlopen(std::filesystem::path(name).filename().c_str(),
                     RTLD_NOW | RTLD_LOCAL);
  if (!handle_)
    throw LibraryError("cannot load runtime library " + name + ": " +
                       dlerror());
#endif
}
Library::~Library() {
#ifdef _WIN32
  if (handle_)
    FreeLibrary(static_cast<HMODULE>(handle_));
#else
  if (handle_)
    dlclose(handle_);
#endif
}
void* Library::Symbol(const char* name) const {
#ifdef _WIN32
  auto* address = reinterpret_cast<void*>(
      GetProcAddress(static_cast<HMODULE>(handle_), name));
  if (!address)
    throw LibraryError(std::string("missing runtime symbol ") + name);
#else
  dlerror();
  auto* address = dlsym(handle_, name);
  if (const auto* error = dlerror())
    throw LibraryError(std::string("missing runtime symbol ") + name + ": " +
                       error);
#endif
  return address;
}
}  // namespace gufo::platform
