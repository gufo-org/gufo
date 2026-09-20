#include <stdexcept>

#include "src/models/qwen_image_21/model.hpp"

namespace gufo::models::qwen_image_21 {
struct Model::Impl {};
Model::Model(const std::filesystem::path&) {
  throw std::runtime_error("Qwen-Image requires HIP");
}
Model::~Model() = default;
Result Model::Generate(const Request&, const CancellationCheck&,
                       const Observer&) {
  throw std::runtime_error("Qwen-Image requires HIP");
}
std::vector<std::uint32_t> Model::Tokenize(std::string_view) const {
  throw std::runtime_error("Qwen-Image requires HIP");
}
}  // namespace gufo::models::qwen_image_21
