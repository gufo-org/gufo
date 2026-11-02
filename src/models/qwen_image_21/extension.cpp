#include <filesystem>
#include <gufo/ext/image.hpp>
#include <gufo/ext/registry.hpp>
#include <string>
#include <string_view>

#include "probe.hpp"
#include "weights.hpp"

namespace gufo::models::qwen_image_21 {

bool ProbeQwenImage21(std::string_view model_root) {
  try {
    const auto path = std::filesystem::path(model_root) / "model_index.json";
    if (!std::filesystem::exists(path)) {
      return false;
    }
    const auto index = ReadJson(path);
    return index.member_str("_class_name") == "QwenImage21Pipeline";
  } catch (...) {
    return false;
  }
}

bool ResolveQwenImage21Artifacts(std::string_view model_root,
                                 std::string* error) {
  try {
    const auto root = std::filesystem::path(model_root);
    const auto index = ReadJson(root / "model_index.json");
    if (index.member_str("_class_name") != "QwenImage21Pipeline") {
      if (error != nullptr) {
        *error = "not a QwenImage21Pipeline in " + root.string();
      }
      return false;
    }
    return true;
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = e.what();
    }
    return false;
  }
}

}  // namespace gufo::models::qwen_image_21

void qwen_image_21_register(gufo::ext::Registry& registry) {
  using namespace gufo::models::qwen_image_21;
  registry.AddImageProvider({
      .id = "qwen-image-21",
      .probe = &ProbeQwenImage21,
      .resolve = &ResolveQwenImage21Artifacts,
      .create = nullptr,
  });
}
