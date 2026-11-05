#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "src/ext/image.hpp"

namespace gufo::models::qwen_image_21 {

[[nodiscard]] bool ProbeQwenImage21(std::string_view model_root);
[[nodiscard]] bool ResolveQwenImage21Artifacts(std::string_view model_root,
                                               std::string* error = nullptr);

#if defined(ENGINE_ENABLE_HIP)
[[nodiscard]] std::unique_ptr<ext::ImageRunner> CreateQwenImage21Runner(
    const std::string& model_root, std::string* error = nullptr);
#endif

}  // namespace gufo::models::qwen_image_21
