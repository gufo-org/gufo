#include <hip/hip_runtime.h>

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/qwen_gpu_executor.hpp"
#include "src/tokenization/qwen_tokenizer.hpp"

namespace {

using strix::tokenization::TokenId;

struct Options {
  std::string model_path;
  std::uint32_t context_tokens{128};
  std::vector<std::size_t> draft_lengths{1, 2, 4, 8, 16};
};

std::uint32_t ParsePositiveU32(std::string_view value,
                               std::string_view option) {
  std::uint32_t parsed = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() ||
      parsed == 0) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return parsed;
}

std::vector<std::size_t> ParseDraftLengths(std::string_view value) {
  std::vector<std::size_t> lengths;
  std::size_t begin = 0;
  while (begin < value.size()) {
    const auto separator = value.find(',', begin);
    const auto end =
        separator == std::string_view::npos ? value.size() : separator;
    const auto length =
        ParsePositiveU32(value.substr(begin, end - begin), "--draft-lengths");
    if (length > strix::hip::kSsmReplayCapacity) {
      throw std::invalid_argument("draft length exceeds replay capacity");
    }
    lengths.push_back(length);
    begin = end + 1;
  }
  if (lengths.empty()) {
    throw std::invalid_argument("--draft-lengths cannot be empty");
  }
  return lengths;
}

Options ParseOptions(std::span<const char* const> arguments) {
  Options options;
  for (std::size_t index = 1; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    if (argument == "--model" && index + 1 < arguments.size()) {
      options.model_path = arguments[++index];
    } else if (argument == "--context" && index + 1 < arguments.size()) {
      options.context_tokens =
          ParsePositiveU32(arguments[++index], "--context");
    } else if (argument == "--draft-lengths" && index + 1 < arguments.size()) {
      options.draft_lengths = ParseDraftLengths(arguments[++index]);
    } else {
      throw std::invalid_argument("unknown or incomplete option: " +
                                  std::string(argument));
    }
  }
  if (options.model_path.empty()) {
    throw std::invalid_argument("--model is required");
  }
  return options;
}

std::vector<TokenId> MakePrompt(std::size_t count) {
  std::vector<TokenId> tokens(count);
  for (std::size_t index = 0; index < count; ++index) {
    tokens[index] = static_cast<TokenId>(100 + (index % 1000));
  }
  return tokens;
}

}  // namespace

int main(int argc, const char* const* argv) {
  try {
    const auto options = ParseOptions({argv, static_cast<std::size_t>(argc)});
    std::string error;
    auto reader_owner =
        strix::core::GgufReader::OpenFile(options.model_path, &error);
    if (!reader_owner) {
      throw std::runtime_error(error);
    }
    const std::shared_ptr<const strix::core::GgufReader> reader(
        std::move(reader_owner));
    auto executor = strix::hip::QwenGpuExecutor::CreateFromGguf(
        reader, &error, options.context_tokens + 32);
    if (!executor) {
      throw std::runtime_error(error);
    }

    const auto prompt = MakePrompt(options.context_tokens);
    executor->Reset();
    TokenId current_token = executor->ForwardPromptBatch(prompt);
    const auto checkpoint_begin = std::chrono::steady_clock::now();
    executor->SaveState(options.context_tokens);
    const auto checkpoint_end = std::chrono::steady_clock::now();

    std::vector<TokenId> replay_inputs;
    std::vector<TokenId> expected_predictions;
    replay_inputs.reserve(strix::hip::kSsmReplayCapacity + 1);
    expected_predictions.reserve(strix::hip::kSsmReplayCapacity + 1);
    replay_inputs.push_back(current_token);
    for (std::size_t offset = 0; offset < strix::hip::kSsmReplayCapacity;
         ++offset) {
      const auto prediction = executor->ForwardToken(
          replay_inputs.back(),
          options.context_tokens + static_cast<std::uint32_t>(offset));
      expected_predictions.push_back(prediction);
      replay_inputs.push_back(prediction);
    }
    HIP_CHECK(hipDeviceSynchronize());

    const double checkpoint_ms = std::chrono::duration<double, std::milli>(
                                     checkpoint_end - checkpoint_begin)
                                     .count();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "checkpoint_ms\t" << checkpoint_ms << '\n';
    std::cout << "draft_length\trollback_replay_ms\tverified\n";

    for (const std::size_t draft_length : options.draft_lengths) {
      const auto begin = std::chrono::steady_clock::now();
      executor->RestoreState();
      for (std::size_t offset = 0; offset < draft_length; ++offset) {
        (void)executor->ForwardToken(
            replay_inputs[offset],
            options.context_tokens + static_cast<std::uint32_t>(offset), false);
      }
      HIP_CHECK(hipDeviceSynchronize());
      const auto end = std::chrono::steady_clock::now();

      const auto verification = executor->ForwardToken(
          replay_inputs[draft_length],
          options.context_tokens + static_cast<std::uint32_t>(draft_length));
      bool verified = false;
      if (draft_length < expected_predictions.size()) {
        verified = verification == expected_predictions[draft_length];
      } else {
        executor->RestoreState();
        for (std::size_t offset = 0; offset < draft_length; ++offset) {
          (void)executor->ForwardToken(
              replay_inputs[offset],
              options.context_tokens + static_cast<std::uint32_t>(offset),
              false);
        }
        const auto repeated_verification = executor->ForwardToken(
            replay_inputs[draft_length],
            options.context_tokens + static_cast<std::uint32_t>(draft_length));
        verified = verification == repeated_verification;
      }
      const double elapsed_ms =
          std::chrono::duration<double, std::milli>(end - begin).count();
      std::cout << draft_length << '\t' << elapsed_ms << '\t'
                << (verified ? "true" : "false") << '\n';
      if (!verified) {
        return 2;
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
