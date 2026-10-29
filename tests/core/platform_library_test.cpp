#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "src/core/platform/library.hpp"

int main(int argc, char** argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("expected fixture library path");
    // CMake supplies forward-slash paths on Windows too. Keep the fixture
    // outside the executable directory so this exercises the absolute fallback.
    gufo::platform::Library library(
        std::filesystem::path(argv[1]).generic_string());
    auto probe =
        reinterpret_cast<int (*)()>(library.Symbol("GufoLibraryProbe"));
    if (probe() != 761)
      throw std::runtime_error("loaded the wrong library or symbol");
    bool missing_symbol = false;
    try {
      (void)library.Symbol("GufoMissingSymbol");
    } catch (const gufo::platform::LibraryError&) {
      missing_symbol = true;
    }
    if (!missing_symbol)
      throw std::runtime_error("missing symbol was accepted");
    bool missing_library = false;
    try {
      gufo::platform::Library missing("gufo-missing-library-761");
    } catch (const gufo::platform::LibraryError&) {
      missing_library = true;
    }
    if (!missing_library)
      throw std::runtime_error("missing library was accepted");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
