#ifndef GUFO_MODELS_QWEN_IMAGE_21_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_WEIGHTS_HPP_

#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::models::qwen_image_21 {

struct Weight {
  std::vector<int> shape;
  const void* data{nullptr};
  std::size_t bytes{0};
  bool bf16{true};
};

json::Value ReadJson(const std::filesystem::path& path);

class Weights {
public:
  explicit Weights(const std::filesystem::path& root);
  const Weight& Get(std::string_view name) const;
  bool Contains(std::string_view name) const;
  const json::Value& VaeConfig() const { return vae_config_; }

private:
  void Load(const std::filesystem::path& file, const std::string& prefix);
  std::vector<std::shared_ptr<void>> mappings_;
  std::map<std::string, Weight, std::less<>> weights_;
  json::Value vae_config_;
};

}  // namespace gufo::models::qwen_image_21
#endif
