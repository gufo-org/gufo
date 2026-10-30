#ifndef GUFO_TESTS_SUPPORT_CONTINUATION_IDENTITY_HPP_
#define GUFO_TESTS_SUPPORT_CONTINUATION_IDENTITY_HPP_

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace identity_test {
inline const std::string artifact(64, 'a');
inline const std::string draft(64, 'b');

inline void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
inline void Golden(std::string name, const std::vector<std::uint8_t>& bytes) {
  const auto path =
      std::filesystem::path(GUFO_IDENTITY_FIXTURES) / (name + ".txt");
  std::ifstream file(path, std::ios::binary);
  Require(file.is_open(), "missing identity golden");
  const std::vector<std::uint8_t> expected{std::istreambuf_iterator<char>(file),
                                           {}};
  if (bytes != expected)
    throw std::runtime_error("identity bytes differ: " + path.string());
}
template<class F>
void Reject(F&& fn, std::string_view message) {
  try {
    fn();
  } catch (const std::invalid_argument& error) {
    Require(error.what() == message, "identity validation diagnostic changed");
    return;
  }
  throw std::runtime_error("invalid fingerprint accepted");
}
inline std::string Text(const std::vector<std::uint8_t>& bytes) {
  return {bytes.begin(), bytes.end()};
}
template<class F>
int Run(F&& fn) {
  try {
    fn();
    std::cout << "continuation identity: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
}  // namespace identity_test
#endif  // GUFO_TESTS_SUPPORT_CONTINUATION_IDENTITY_HPP_
