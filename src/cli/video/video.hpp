#ifndef STRIX_CLI_VIDEO_HPP_
#define STRIX_CLI_VIDEO_HPP_

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "src/models/minimax_h3/generation.hpp"

namespace strix::cli {

struct VideoCliOptions {
  minimax_h3::GenerationRequest request;
  std::filesystem::path parameters_path;
  std::filesystem::path telemetry_path;
  bool profile{false};
};

[[nodiscard]] std::filesystem::path DefaultH3SourceManifest();
void PrintVideoHelp(std::string_view program_name);
[[nodiscard]] std::optional<VideoCliOptions> ParseVideoOptions(
    std::span<const char* const> args, std::string* error = nullptr);
[[nodiscard]] int RunVideo(std::span<const char* const> args);

}  // namespace strix::cli

#endif  // STRIX_CLI_VIDEO_HPP_
