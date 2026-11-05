#include "src/models/qwen38_flash_next/projection.hpp"

#include <cmath>
#include <string>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

bool Fail(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
  return false;
}

}  // namespace

bool LoadControlVector(const std::string& path, std::uint32_t num_layers,
                       std::uint32_t hidden, std::string_view expected_hint,
                       ProjectionVector* out, std::string* error_msg,
                       std::string* warning) {
  if (out == nullptr) {
    return Fail(error_msg, "projection output is null");
  }
  std::string open_error;
  auto reader = core::GgufReader::OpenFile(path, &open_error);
  if (reader == nullptr) {
    return Fail(error_msg,
                "cannot open projection vector " + path + ": " + open_error);
  }
  // Strata warns only when the file names a different source model; a file
  // without the key (ours) loads silently.
  if (const auto hint = reader->GetMetadataString("controlvector.model_hint");
      hint.has_value() && *hint != expected_hint && warning != nullptr) {
    *warning = "projection vector " + path + " declares model_hint '" +
               std::string(*hint) + "' but this model is '" +
               std::string(expected_hint) + "'";
  }
  out->directions.assign(static_cast<std::size_t>(num_layers) * hidden, 0.0F);
  out->scales.assign(num_layers, 0.0F);
  std::uint32_t found = 0;
  for (std::uint32_t il = 1; il < num_layers; ++il) {
    const std::string name = "direction." + std::to_string(il);
    const core::GgufTensorInfo* info = reader->FindTensor(name);
    if (info == nullptr) {
      continue;
    }
    if (info->type != core::GgmlType::kF32 || info->ElementCount() != hidden) {
      return Fail(error_msg, name + " must be F32 with " +
                                 std::to_string(hidden) + " elements");
    }
    const auto* d = static_cast<const float*>(info->data);
    double square_norm = 0.0;
    for (std::uint32_t i = 0; i < hidden; ++i) {
      square_norm += static_cast<double>(d[i]) * static_cast<double>(d[i]);
    }
    const double norm = std::sqrt(square_norm);
    if (norm > 0.0) {
      float* row =
          out->directions.data() + static_cast<std::size_t>(il) * hidden;
      const float inverse = static_cast<float>(1.0 / norm);
      for (std::uint32_t i = 0; i < hidden; ++i) {
        row[i] = d[i] * inverse;
      }
      out->scales[il] = static_cast<float>(norm);
      ++found;
    }
  }
  if (found == 0) {
    return Fail(error_msg, "projection vector " + path +
                               " carries no usable direction.N tensors");
  }
  out->num_layers = num_layers;
  out->hidden = hidden;
  out->first_layer = 1;
  out->last_layer = num_layers - 1;
  return true;
}

}  // namespace gufo::models::qwen38_flash_next
