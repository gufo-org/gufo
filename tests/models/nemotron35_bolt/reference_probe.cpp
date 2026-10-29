// Runs the float32 reference over a token sequence and reports the next-token
// distribution at every position. Used to pin the operator semantics against
// llama.cpp logits (tools/llama_parity/llama_logits.cpp) and, later, against
// the ROCm runtime.
//
//   nemotron35_bolt_reference_probe --model MODEL.gguf --tokens 1,2,3
//       [--generate N] [--dump logits.bin] [--top 5]
//
// --dump writes float32 logits for every prompt position (and generated
// position) back to back; the row width is the vocabulary size.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/nemotron35_bolt/reference.hpp"
#include "src/models/nemotron35_bolt/weights.hpp"

namespace nb = gufo::models::nemotron35_bolt;

namespace {

std::vector<std::int32_t> ParseTokens(const std::string& text) {
  std::vector<std::int32_t> out;
  std::size_t start = 0;
  while (start < text.size()) {
    const std::size_t end = text.find(',', start);
    out.push_back(std::atoi(text.substr(start, end - start).c_str()));
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return out;
}

std::uint32_t Argmax(const std::vector<float>& logits) {
  return static_cast<std::uint32_t>(
      std::max_element(logits.begin(), logits.end()) - logits.begin());
}

void PrintTop(const std::vector<float>& logits, std::uint32_t top) {
  std::vector<std::uint32_t> order(logits.size());
  std::iota(order.begin(), order.end(), 0U);
  std::partial_sort(
      order.begin(), order.begin() + top, order.end(),
      [&](std::uint32_t a, std::uint32_t b) { return logits[a] > logits[b]; });
  const float max_logit = logits[order[0]];
  double denom = 0.0;
  for (float v : logits) {
    denom += std::exp(static_cast<double>(v - max_logit));
  }
  for (std::uint32_t i = 0; i < top; ++i) {
    const std::uint32_t id = order[i];
    std::printf("   %6u  logit=%9.4f  p=%.5f\n", id, logits[id],
                std::exp(static_cast<double>(logits[id] - max_logit)) / denom);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string model;
  std::string tokens_arg;
  std::string dump;
  std::uint32_t generate = 0;
  std::uint32_t top = 5;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--model") {
      model = value();
    } else if (arg == "--tokens") {
      tokens_arg = value();
    } else if (arg == "--dump") {
      dump = value();
    } else if (arg == "--generate") {
      generate = static_cast<std::uint32_t>(std::stoul(value()));
    } else if (arg == "--top") {
      top = static_cast<std::uint32_t>(std::stoul(value()));
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  const auto tokens = ParseTokens(tokens_arg);
  if (model.empty() || tokens.empty()) {
    std::fprintf(stderr,
                 "usage: %s --model MODEL.gguf --tokens 1,2,3 [--generate N] "
                 "[--dump logits.bin] [--top N]\n",
                 argv[0]);
    return 2;
  }

  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model, &error);
  if (!reader) {
    std::fprintf(stderr, "open %s: %s\n", model.c_str(), error.c_str());
    return 1;
  }
  auto weights = nb::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    std::fprintf(stderr, "bind: %s\n", error.c_str());
    return 1;
  }
  const auto vocab = weights->config.vocab_size;
  nb::ReferenceModel ref(*weights,
                         static_cast<std::uint32_t>(tokens.size()) + generate);
  top = std::min(top, vocab);

  std::ofstream out;
  if (!dump.empty()) {
    out.open(dump, std::ios::binary);
    if (!out) {
      std::fprintf(stderr, "cannot write %s\n", dump.c_str());
      return 1;
    }
  }
  std::vector<float> logits(vocab);
  std::int32_t next = tokens.front();
  const std::size_t steps = tokens.size() + generate;
  for (std::size_t pos = 0; pos < steps; ++pos) {
    const std::int32_t token = pos < tokens.size() ? tokens[pos] : next;
    const auto start = std::chrono::steady_clock::now();
    if (!ref.Step(token, logits, &error)) {
      std::fprintf(stderr, "step %zu: %s\n", pos, error.c_str());
      return 1;
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    next = static_cast<std::int32_t>(Argmax(logits));
    std::printf("pos %zu token %d (%.2f s) -> argmax %d\n", pos, token, seconds,
                next);
    PrintTop(logits, top);
    if (out) {
      out.write(reinterpret_cast<const char*>(logits.data()),
                static_cast<std::streamsize>(logits.size() * sizeof(float)));
    }
  }
  return 0;
}
