#ifndef GUFO_CLI_VIDEO_HPP_
#define GUFO_CLI_VIDEO_HPP_

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "src/ext/video.hpp"

namespace gufo::cli {

struct VideoCliOptions {
  ext::VideoRequest request;
  std::filesystem::path parameters_path;
  std::filesystem::path telemetry_path;
  bool profile{false};
};

[[nodiscard]] std::filesystem::path DefaultH3SourceManifest();
void PrintVideoHelp(std::string_view program_name);
[[nodiscard]] std::optional<VideoCliOptions> ParseVideoOptions(
    std::span<const char* const> args, std::string* error = nullptr);
[[nodiscard]] int RunVideo(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_VIDEO_HPP_
