#ifndef GUFO_TESTS_MODELS_GEMMA4_LOGIT_FILE_HPP_
#define GUFO_TESTS_MODELS_GEMMA4_LOGIT_FILE_HPP_

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "tests/models/gemma4/check.hpp"

namespace gemma4_test {

/// Logit rows in the format of tools/gemma4/llama_logits.cpp.
struct LogitFile {
  std::uint32_t vocab{0};
  std::vector<std::uint32_t> positions;
  std::vector<float> logits;  ///< [positions][vocab]
};

inline void WriteLogits(const std::string& path, const LogitFile& file) {
  std::ofstream out(path, std::ios::binary);
  const std::uint32_t header[4] = {
      0x474C3447U, 1U, static_cast<std::uint32_t>(file.positions.size()),
      file.vocab};
  out.write(reinterpret_cast<const char*>(header), sizeof(header));
  out.write(reinterpret_cast<const char*>(file.positions.data()),
            static_cast<std::streamsize>(file.positions.size() * 4));
  out.write(reinterpret_cast<const char*>(file.logits.data()),
            static_cast<std::streamsize>(file.logits.size() * 4));
  Require(out.good(), "cannot write " + path);
}

inline void WriteTokens(const std::string& path,
                        const std::vector<std::int32_t>& tokens) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(tokens.data()),
            static_cast<std::streamsize>(tokens.size() * 4));
  Require(out.good(), "cannot write " + path);
}

}  // namespace gemma4_test

#endif  // GUFO_TESTS_MODELS_GEMMA4_LOGIT_FILE_HPP_
