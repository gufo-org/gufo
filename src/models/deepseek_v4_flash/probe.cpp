#include "src/models/deepseek_v4_flash/probe.hpp"

#include <filesystem>
#include <string>

namespace gufo::models::deepseek_v4_flash {

bool ProbeDeepSeekGguf(const core::GgufReader& reader) {
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  return architecture == "deepseek4";
}

bool ProbeDeepSeek(std::string_view model_root) {
  try {
    std::filesystem::path p(model_root);
    if (!std::filesystem::exists(p)) {
      return false;
    }
    if (std::filesystem::is_regular_file(p)) {
      std::string err;
      auto reader = core::GgufReader::OpenFile(p.string(), &err);
      if (!reader)
        return false;
      return ProbeDeepSeekGguf(*reader);
    }
    return false;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::deepseek_v4_flash
