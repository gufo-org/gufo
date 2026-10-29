#ifndef GUFO_CORE_PLATFORM_LIBRARY_HPP_
#define GUFO_CORE_PLATFORM_LIBRARY_HPP_

#include <stdexcept>
#include <string>

namespace gufo::platform {
class LibraryError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// Loads one runtime library on demand. The caller keeps this object alive
/// while using any symbol obtained from it. Failures throw std::runtime_error.
class Library {
public:
  explicit Library(const std::string& name);
  ~Library();
  Library(const Library&) = delete;
  Library& operator=(const Library&) = delete;
  [[nodiscard]] void* Symbol(const char* name) const;

private:
  void* handle_{nullptr};
};
}  // namespace gufo::platform
#endif
