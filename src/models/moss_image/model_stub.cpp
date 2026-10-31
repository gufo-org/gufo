#include <stdexcept>

#include "src/models/moss_image/model.hpp"

namespace gufo::models::moss_image {

struct Model::Impl {};

Model::Model(const std::filesystem::path&, Options) {
  throw std::runtime_error("MOSS-Image requires the HIP build");
}
Model::~Model() = default;

Result Model::Generate(const Request&, const CancellationCheck&,
                       const Observer&) {
  throw std::runtime_error("MOSS-Image requires the HIP build");
}

}  // namespace gufo::models::moss_image
