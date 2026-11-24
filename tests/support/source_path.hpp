#ifndef GUFO_TESTS_SUPPORT_SOURCE_PATH_HPP_
#define GUFO_TESTS_SUPPORT_SOURCE_PATH_HPP_

// Source-tree fixture paths are compiled into tests as absolute paths. A test
// bundle copied off the build host (tools/ci/ctest_bundle.py) sets
// GUFO_TEST_SOURCE_DIR to its copy of those files; in a build tree the
// compiled path is used unchanged. Link gufo_test_source_path to use this.

#include <cstdlib>
#include <string>
#include <string_view>

namespace gufo::testing {

inline std::string SourcePath(std::string_view compiled) {
  constexpr std::string_view built = GUFO_TEST_BUILD_SOURCE_DIR;
  const char* root = std::getenv("GUFO_TEST_SOURCE_DIR");
  if (root == nullptr || *root == '\0' ||
      compiled.substr(0, built.size()) != built ||
      (compiled.size() > built.size() && compiled[built.size()] != '/')) {
    return std::string(compiled);
  }
  return std::string(root) + std::string(compiled.substr(built.size()));
}

}  // namespace gufo::testing

#endif  // GUFO_TESTS_SUPPORT_SOURCE_PATH_HPP_
