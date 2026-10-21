// Full-vocabulary AR teacher-forcing logits for offline KL comparisons.
// Usage: qwen_kl_probe MODEL.gguf TOKENS.i32 OUTDIR
// Input: little-endian int32 [N][T], T=256 unless KL_T is set.
// Output: OUTDIR/NN.f32, little-endian float32 [T][vocab]. Row p predicts
// the token after input position p. Reset state for every sequence; no
// sampling, prompt batching, or speculative decoding. KL_LIMIT bounds sequences
// processed. KL_MODE=prefill emits only the final-position row per sequence
// using batched prefill; the default KL_MODE=ar emits every AR row.

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {

static_assert(std::endian::native == std::endian::little);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

std::size_t PositiveEnv(const char* name, std::size_t fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr)
    return fallback;
  const std::string_view text(raw);
  std::size_t value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value == 0)
    throw std::runtime_error(std::string(name) +
                             " must be a positive decimal integer");
  return value;
}

void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " MODEL.gguf TOKENS.i32 OUTDIR\n";
    return 2;
  }
  std::filesystem::path partial;
  try {
    const char* mode_env = std::getenv("KL_MODE");
    const std::string_view mode = mode_env == nullptr ? "ar" : mode_env;
    Require(mode == "ar" || mode == "prefill", "KL_MODE must be ar or prefill");
    const bool prefill = mode == "prefill";
    const std::size_t length = PositiveEnv("KL_T", 256);
    const std::size_t output_rows = prefill ? 1 : length;
    const std::size_t limit =
        PositiveEnv("KL_LIMIT", std::numeric_limits<std::size_t>::max());
    Require(length <= std::numeric_limits<std::uint32_t>::max() &&
                length <= std::numeric_limits<std::size_t>::max() /
                              sizeof(std::int32_t),
            "KL_T exceeds the supported position or input size range");
    const std::size_t sequence_bytes = length * sizeof(std::int32_t);
    std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
    Require(input.is_open(), "cannot open token file: " + std::string(argv[2]));
    const auto end = input.tellg();
    Require(end > 0, "token file is empty or its size cannot be read");
    const auto bytes = static_cast<std::uintmax_t>(end);
    Require(bytes <= std::numeric_limits<std::size_t>::max() &&
                bytes <= static_cast<std::uintmax_t>(
                             std::numeric_limits<std::streamsize>::max()) &&
                bytes % sequence_bytes == 0,
            "token file must contain complete nonempty [N][KL_T] int32 rows");
    std::vector<std::int32_t> tokens(bytes / sizeof(std::int32_t));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(tokens.data()),
               static_cast<std::streamsize>(bytes));
    Require(input.good(), "failed to read complete token file");
    Require(input.peek() == std::char_traits<char>::eof() && !input.bad(),
            "token file changed size or failed while reading");
    const std::size_t sequences = tokens.size() / length;
    const std::size_t selected = std::min(sequences, limit);

    std::string error;
    std::shared_ptr<const gufo::core::GgufReader> reader =
        gufo::core::GgufReader::OpenFile(argv[1], &error);
    Require(reader != nullptr, "open model failed: " + error);
    const auto config = reader->ExtractModelConfig(&error);
    Require(config.has_value(), "model config failed: " + error);
    const std::size_t vocab = config->vocab_size;
    Require(length <= config->context_length,
            "KL_T exceeds the model context length");
    for (std::size_t i = 0; i < tokens.size(); ++i)
      Require(tokens[i] >= 0 && static_cast<std::uint32_t>(tokens[i]) < vocab,
              "token out of vocabulary at flat index " + std::to_string(i));
    Require(vocab <= static_cast<std::size_t>(
                         std::numeric_limits<std::streamsize>::max()) /
                         sizeof(float) &&
                length <= std::numeric_limits<std::uintmax_t>::max() /
                              sizeof(float) / vocab,
            "logit output size exceeds supported range");
    const auto row_bytes = static_cast<std::streamsize>(vocab * sizeof(float));
    const auto output_bytes =
        std::uintmax_t{output_rows} * vocab * sizeof(float);
    const std::filesystem::path directory(argv[3]);
    std::filesystem::create_directories(directory);
    Require(std::filesystem::is_directory(directory),
            "OUTDIR is not a directory");
    std::cout << "mode=" << mode << " reset_per_sequence=true dtype=float32_le"
              << " model=" << argv[1] << " tokens=" << argv[2]
              << " total_sequences=" << sequences
              << " selected_sequences=" << selected << " T=" << length
              << " vocab=" << vocab << " output_rows=" << output_rows
              << " bytes_per_output=" << output_bytes << '\n'
              << std::flush;

    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    Require(model != nullptr, "GPU model creation failed: " + error);
    auto executor = gufo::hip::QwenGpuExecutor::Create(
        model, &error, static_cast<std::uint32_t>(length));
    Require(executor != nullptr, "GPU executor creation failed: " + error);
    for (std::size_t sequence = 0; sequence < selected; ++sequence) {
      const auto started = std::chrono::steady_clock::now();
      std::ostringstream name;
      name << std::setfill('0') << std::setw(2) << sequence << ".f32";
      const auto destination = directory / name.str();
      partial = destination;
      partial += ".partial";
      std::ofstream output(partial, std::ios::binary | std::ios::trunc);
      Require(output.is_open(), "cannot open output: " + partial.string());
      executor->Reset();
      if (prefill) {
        const auto row = std::span(tokens).subspan(sequence * length, length);
        const std::vector<gufo::tokenization::TokenId> prompt(row.begin(),
                                                              row.end());
        (void)executor->ForwardPromptBatch(prompt, 0, true);
      }
      for (std::size_t position = 0; position < output_rows; ++position) {
        if (!prefill) {
          (void)executor->ForwardToken(
              static_cast<gufo::tokenization::TokenId>(
                  tokens[sequence * length + position]),
              static_cast<std::uint32_t>(position), true);
        }
        const auto logits = executor->CopyLastLogits();
        Require(logits.size() == vocab, "incorrect logit row size");
        Require(std::all_of(logits.begin(), logits.end(),
                            [](float value) { return std::isfinite(value); }),
                "non-finite logits at sequence " + std::to_string(sequence) +
                    " position " + std::to_string(position));
        output.write(reinterpret_cast<const char*>(logits.data()), row_bytes);
        Require(output.good(), "failed writing output: " + partial.string());
      }
      output.close();
      Require(!output.fail(), "failed closing output: " + partial.string());
      Require(std::filesystem::file_size(partial) == output_bytes,
              "incorrect output file size: " + partial.string());
      std::filesystem::rename(partial, destination);
      partial.clear();
      const double seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
      std::cout << '[' << sequence + 1 << '/' << selected << "] " << destination
                << " seconds=" << seconds << '\n'
                << std::flush;
    }
    return 0;
  } catch (const std::exception& error) {
    if (!partial.empty()) {
      std::error_code ignored;
      std::filesystem::remove(partial, ignored);
    }
    std::cerr << "qwen_kl_probe: " << error.what() << '\n';
    return 1;
  }
}
