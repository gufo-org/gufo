#include "src/models/qwen38_flash_next/projection.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <span>
#include <string>
#include <system_error>

#include "gufo/ext/gguf_reader.hpp"
#include "gufo/ext/sha256.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

constexpr std::string_view kDirectionPrefix = "direction.";
// Norms this close to one are unit directions written in F32.
constexpr double kUnitNormTolerance = 1e-3;

bool Fail(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
  return false;
}

void Warn(std::string* warning, const std::string& message) {
  if (warning == nullptr) {
    return;
  }
  if (!warning->empty()) {
    *warning += "; ";
  }
  *warning += message;
}

bool ParseUint(std::string_view text, std::uint32_t* value) {
  const auto* end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(text.data(), end, *value);
  return !text.empty() && ec == std::errc{} && ptr == end;
}

template<class T>
void HashValues(crypto::Sha256Hasher& hasher, std::span<const T> values) {
  hasher.Update({reinterpret_cast<const std::uint8_t*>(values.data()),
                 values.size_bytes()});
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
  // Warn only when the file names a different source model; a file without
  // the key loads silently.
  if (const auto hint = reader->GetMetadataString("controlvector.model_hint");
      hint.has_value() && *hint != expected_hint) {
    Warn(warning, "projection vector " + path + " declares model_hint '" +
                      std::string(*hint) + "' but this model is '" +
                      std::string(expected_hint) + "'");
  }
  for (const auto& tensor : reader->GetTensors()) {
    if (!tensor.name.starts_with(kDirectionPrefix)) {
      continue;
    }
    std::uint32_t il = 0;
    if (!ParseUint(tensor.name.substr(kDirectionPrefix.size()), &il) ||
        il == 0 || il >= num_layers) {
      Warn(warning, "ignoring " + std::string(tensor.name) +
                        ": this model steers direction.1..direction." +
                        std::to_string(num_layers > 0 ? num_layers - 1 : 0));
    }
  }
  out->directions.assign(static_cast<std::size_t>(num_layers) * hidden, 0.0F);
  out->scales.assign(num_layers, 0.0F);
  std::uint32_t found = 0;
  std::string non_unit;
  for (std::uint32_t il = 1; il < num_layers; ++il) {
    const std::string name = std::string(kDirectionPrefix) + std::to_string(il);
    const core::GgufTensorInfo* info = reader->FindTensor(name);
    if (info == nullptr) {
      continue;
    }
    if (info->type != core::GgmlType::kF32 || info->ElementCount() != hidden) {
      return Fail(error_msg, name + " must be F32 with " +
                                 std::to_string(hidden) + " elements");
    }
    // GGUF only aligns tensor data to general.alignment; copy before reading
    // it as floats.
    float* row = out->directions.data() + static_cast<std::size_t>(il) * hidden;
    std::memcpy(row, info->data,
                static_cast<std::size_t>(hidden) * sizeof(float));
    // A non-finite element marks a corrupt file; steering with it would
    // turn every projected residual into NaN.
    if (!std::all_of(row, row + hidden,
                     [](float x) { return std::isfinite(x); })) {
      return Fail(error_msg, name + " contains non-finite values");
    }
    double square_norm = 0.0;
    for (std::uint32_t i = 0; i < hidden; ++i) {
      square_norm += static_cast<double>(row[i]) * static_cast<double>(row[i]);
    }
    const double norm = std::sqrt(square_norm);
    if (norm == 0.0) {
      Warn(warning, "ignoring " + name + ": zero direction");
      continue;
    }
    const auto scale = static_cast<float>(norm);
    if (!std::isfinite(scale)) {
      return Fail(error_msg, name + " has a norm beyond the F32 range");
    }
    // Divide in double: the reciprocal of a tiny norm overflows F32.
    for (std::uint32_t i = 0; i < hidden; ++i) {
      row[i] = static_cast<float>(static_cast<double>(row[i]) / norm);
    }
    out->scales[il] = scale;
    ++found;
    if (std::abs(norm - 1.0) > kUnitNormTolerance) {
      non_unit += (non_unit.empty() ? "" : ", ") + std::to_string(il);
    }
  }
  if (found == 0) {
    return Fail(error_msg, "projection vector " + path +
                               " carries no usable direction.N tensors");
  }
  if (!non_unit.empty()) {
    Warn(warning, "projection vector " + path +
                      " has non-unit directions at layers " + non_unit +
                      "; their norm scales the update, so it no longer removes "
                      "exactly the component along the direction");
  }
  out->num_layers = num_layers;
  out->hidden = hidden;
  out->first_layer = 1;
  out->last_layer = num_layers - 1;
  return true;
}

bool SelectProjectionLayers(std::string_view spec, ProjectionVector* projection,
                            std::string* error_msg) {
  if (projection == nullptr || projection->Empty()) {
    return Fail(error_msg, "no projection vector is loaded");
  }
  const std::uint32_t max_layer = projection->num_layers - 1;
  if (spec.empty()) {
    projection->first_layer = 1;
    projection->last_layer = max_layer;
    return true;
  }
  const auto dash = spec.find('-');
  std::uint32_t lo = 0;
  std::uint32_t hi = 0;
  if (dash == std::string_view::npos || !ParseUint(spec.substr(0, dash), &lo) ||
      !ParseUint(spec.substr(dash + 1), &hi)) {
    return Fail(error_msg,
                "projection layers '" + std::string(spec) + "' must be LO-HI");
  }
  if (lo < 1 || lo > hi || hi > max_layer) {
    return Fail(error_msg, "projection layers '" + std::string(spec) +
                               "' must satisfy 1 <= LO <= HI <= " +
                               std::to_string(max_layer));
  }
  projection->first_layer = lo;
  projection->last_layer = hi;
  return true;
}

std::string ProjectionFingerprint(const ProjectionVector& projection) {
  crypto::Sha256Hasher hasher;
  const std::uint32_t shape[] = {projection.num_layers, projection.hidden,
                                 projection.first_layer, projection.last_layer};
  HashValues(hasher, std::span<const std::uint32_t>(shape));
  HashValues(hasher, std::span<const float>(projection.scales));
  HashValues(hasher, std::span<const float>(projection.directions));
  return hasher.FinishHex();
}

}  // namespace gufo::models::qwen38_flash_next
