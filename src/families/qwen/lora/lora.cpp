#include "src/families/qwen/lora/lora.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

#include "gufo/ext/experimental/ggml_dequant.hpp"
#include "gufo/ext/sha256.hpp"

// The merge must reproduce offline merges bit for bit: this file is compiled
// with -ffp-contract=off (CMakeLists.txt) so no multiply-add is fused.

namespace gufo::models::qwen::lora {
namespace {

constexpr std::size_t kQ8Block = 32;
constexpr std::size_t kQ8BlockBytes = 34;

std::string Shape(std::span<const std::uint64_t> dimensions) {
  std::string text = "[";
  for (std::size_t i = 0; i < dimensions.size(); ++i) {
    text += (i == 0 ? "" : ", ") + std::to_string(dimensions[i]);
  }
  return text + "]";
}

std::string FloatText(float value) {
  char text[64];
  std::snprintf(text, sizeof(text), "%.9g", static_cast<double>(value));
  return text;
}

/// Bytes per element of a factor format the merge reads, or 0.
std::size_t FactorBytes(core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kF32:
      return 4;
    case core::GgmlType::kF16:
    case core::GgmlType::kBF16:
      return 2;
    default:
      return 0;
  }
}

/// Widens a factor of a format FactorBytes accepts to float32.
std::vector<float> Widen(const core::GgufTensorInfo& info) {
  const std::size_t count = info.ElementCount();
  std::vector<float> values(count);
  if (info.type == core::GgmlType::kF32) {
    std::memcpy(values.data(), info.data, count * sizeof(float));
    return values;
  }
  const auto* bits = static_cast<const std::uint8_t*>(info.data);
  for (std::size_t i = 0; i < count; ++i) {
    std::uint16_t half = 0;
    std::memcpy(&half, bits + i * 2, sizeof(half));
    values[i] =
        info.type == core::GgmlType::kF16
            ? quant::Fp16ToFloat(half)
            : std::bit_cast<float>(static_cast<std::uint32_t>(half) << 16);
  }
  return values;
}

}  // namespace

bool ResolveLoraOptions(const ext::ParsedOptions& options,
                        LoraSelection* selection, std::string* error) {
  const auto fail = [&](std::string message) {
    if (error != nullptr) {
      *error = std::move(message);
    }
    return false;
  };
  *selection = {};
  const auto* path = ext::FindOption<std::string>(options, "--lora");
  const auto* scale = ext::FindOption<double>(options, "--lora-scale");
  if (path == nullptr) {
    return scale == nullptr || fail("--lora-scale requires --lora");
  }
  if (path->empty()) {
    return fail("--lora requires a path");
  }
  if (scale != nullptr) {
    const auto narrowed = static_cast<float>(*scale);
    if (!std::isfinite(*scale) || !std::isfinite(narrowed)) {
      return fail("--lora-scale must be a finite float32 number");
    }
    selection->scale = narrowed;
  }
  selection->path = *path;
  return true;
}

std::size_t LoraMatrix::RowBytes() const noexcept {
  return quant::EncodedSizeBytes(type, cols);
}

std::unique_ptr<LoraAdapter> LoraAdapter::Open(
    const std::filesystem::path& path, float scale,
    std::string_view architecture, std::span<const LoraMatrix> targets,
    std::string_view expected_targets, std::string* error) {
  std::string open_error;
  auto reader = core::GgufReader::OpenFile(path, &open_error);
  if (!reader) {
    if (error != nullptr) {
      *error = "LoRA adapter " + path.string() + ": " + open_error;
    }
    return nullptr;
  }
  return Open(std::move(reader), path.string(), scale, architecture, targets,
              expected_targets, error);
}

std::unique_ptr<LoraAdapter> LoraAdapter::Open(
    std::unique_ptr<core::GgufReader> reader, std::string label, float scale,
    std::string_view architecture, std::span<const LoraMatrix> targets,
    std::string_view expected_targets, std::string* error) {
  const auto fail = [&](const std::string& message) {
    if (error != nullptr) {
      *error = "LoRA adapter " + label + ": " + message;
    }
    return nullptr;
  };
  if (!std::isfinite(scale)) {
    return fail("scale " + FloatText(scale) + " is not finite");
  }
  const auto text = [&](std::string_view key) {
    const auto value = reader->GetMetadataString(key);
    return value ? "'" + std::string(*value) + "'" : std::string("missing");
  };
  if (reader->GetMetadataString("general.architecture") != architecture) {
    return fail("general.architecture is " + text("general.architecture") +
                ", expected '" + std::string(architecture) + "'");
  }
  if (reader->GetMetadataString("general.type") != "adapter") {
    return fail("general.type is " + text("general.type") +
                ", expected 'adapter'");
  }
  if (reader->GetMetadataString("adapter.type") != "lora") {
    return fail("adapter.type is " + text("adapter.type") +
                ", expected 'lora'");
  }
  if (reader->FindMetadata("adapter.alora.invocation_tokens") != nullptr) {
    return fail(
        "is an activated LoRA (adapter.alora.invocation_tokens), which a "
        "load-time merge cannot apply");
  }
  const auto alpha = reader->GetMetadataFloat32("adapter.lora.alpha");
  if (!alpha) {
    return fail("adapter.lora.alpha is missing, expected a float32");
  }
  if (!std::isfinite(*alpha) || *alpha <= 0.0F) {
    return fail("adapter.lora.alpha is " + FloatText(*alpha) +
                ", expected a positive finite value");
  }

  std::map<std::string_view, const LoraMatrix*> by_name;
  for (const auto& target : targets) {
    by_name.emplace(target.name, &target);
  }
  struct Pair {
    const core::GgufTensorInfo* a{nullptr};
    const core::GgufTensorInfo* b{nullptr};
  };
  std::map<std::string, Pair, std::less<>> pairs;
  for (const auto& info : reader->GetTensors()) {
    const std::string_view name = info.name;
    const bool is_a = name.ends_with(".lora_a");
    if (!is_a && !name.ends_with(".lora_b")) {
      return fail("tensor " + std::string(name) +
                  " has an unknown suffix, expected .lora_a or .lora_b");
    }
    // The reader rejects repeated names, so each half is seen once.
    auto& pair = pairs[std::string(name.substr(0, name.size() - 7))];
    (is_a ? pair.a : pair.b) = &info;
  }
  if (pairs.empty()) {
    return fail("contains no LoRA tensors");
  }

  std::unique_ptr<LoraAdapter> adapter(new LoraAdapter());
  for (const auto& [base, pair] : pairs) {
    if (pair.a == nullptr || pair.b == nullptr) {
      return fail("tensor " + base +
                  (pair.a == nullptr ? ".lora_a" : ".lora_b") +
                  " is missing, found only " + base +
                  (pair.a == nullptr ? ".lora_b" : ".lora_a"));
    }
    const auto found = by_name.find(base);
    if (found == by_name.end()) {
      return fail("target " + base +
                  " is not a patchable trunk tensor, expected " +
                  std::string(expected_targets));
    }
    const LoraMatrix& target = *found->second;
    if (target.type != core::GgmlType::kQ8_0) {
      return fail("target " + base + " is " +
                  std::string(core::ToString(target.type)) +
                  " in the model, expected Q8_0");
    }
    for (const auto* info : {pair.a, pair.b}) {
      if (FactorBytes(info->type) == 0) {
        return fail("tensor " + std::string(info->name) + " is " +
                    std::string(core::ToString(info->type)) +
                    ", expected F32, F16 or BF16");
      }
    }
    const auto& da = pair.a->dimensions;
    const auto& db = pair.b->dimensions;
    if (da.size() != 2 || da[0] != target.cols || da[1] == 0 ||
        da[1] > std::numeric_limits<std::uint32_t>::max()) {
      return fail("tensor " + base + ".lora_a has shape " + Shape(da) +
                  ", expected [" + std::to_string(target.cols) + ", rank]");
    }
    if (db.size() != 2 || db[1] != target.rows || db[0] == 0) {
      return fail("tensor " + base + ".lora_b has shape " + Shape(db) +
                  ", expected [rank, " + std::to_string(target.rows) + "]");
    }
    if (da[1] != db[0]) {
      return fail("tensor " + base + " has rank " + std::to_string(da[1]) +
                  " in .lora_a but " + std::to_string(db[0]) + " in .lora_b");
    }
    LoraPatch patch{target, static_cast<std::uint32_t>(da[1]), {}, {}};
    for (const auto* info : {pair.a, pair.b}) {
      const std::uint64_t count = info->ElementCount();
      const std::uint64_t bytes = count * FactorBytes(info->type);
      if (!info->sized || info->data == nullptr || info->size_bytes != bytes) {
        return fail("tensor " + std::string(info->name) + " has " +
                    std::to_string(info->size_bytes) + " bytes, expected " +
                    std::to_string(bytes));
      }
      auto values = Widen(*info);
      for (std::uint64_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
          return fail("tensor " + std::string(info->name) +
                      " has a non-finite value at element " +
                      std::to_string(i));
        }
      }
      (info == pair.a ? patch.a : patch.b) = std::move(values);
    }
    adapter->patches_.emplace(base, std::move(patch));
  }

  const auto regions = reader->GetMappedRegions();
  if (regions.size() != 1) {
    return fail("must be a single GGUF file");
  }
  adapter->sha256_ = crypto::Sha256Hex(
      {static_cast<const std::uint8_t*>(regions[0].data), regions[0].size});
  adapter->label_ = std::move(label);
  adapter->scale_ = scale;
  adapter->alpha_ = *alpha;
  return adapter;
}

const LoraPatch* LoraAdapter::Find(std::string_view name) const {
  const auto found = patches_.find(name);
  return found == patches_.end() ? nullptr : &found->second;
}

float LoraAdapter::Multiplier(const LoraPatch& patch) const noexcept {
  return scale_ * (alpha_ / static_cast<float>(patch.rank));
}

std::string LoraAdapter::CacheIdentity() const {
  char bits[16];
  std::snprintf(bits, sizeof(bits), "%08x",
                static_cast<unsigned>(std::bit_cast<std::uint32_t>(scale_)));
  return "lora;merge=" + std::string(kLoraMergeAlgorithm) +
         ";sha256=" + sha256_ + ";scale_f32=0x" + bits;
}

bool LoraAdapter::MergeRows(const LoraPatch& patch, std::uint64_t first,
                            std::uint64_t count, std::uint8_t* out,
                            std::string* error) const {
  const LoraMatrix& t = patch.target;
  const std::size_t cols = t.cols;
  const std::size_t row_bytes = t.RowBytes();
  const float multiplier = Multiplier(patch);
  const auto* source = static_cast<const std::uint8_t*>(t.data);
  const float* factor_a = patch.a.data();
  const float* factor_b = patch.b.data();
  // Rows are independent, so threads change no result bit.
  const auto rows = static_cast<std::int64_t>(count);
  std::int64_t failed = rows;
#pragma omp parallel
  {
    std::vector<float> row(cols);
    std::vector<float> delta(cols);
#pragma omp for schedule(static)
    for (std::int64_t i = 0; i < rows; ++i) {
      const std::uint64_t r = first + static_cast<std::uint64_t>(i);
      quant::DequantizeQ8_0(source + r * row_bytes, row.data(), cols);
      // delta = sum_k B[r][k] * A[k], in rank order, then times the
      // multiplier; each product and sum rounds to float32 on its own.
      const float* b = factor_b + r * patch.rank;
      for (std::size_t c = 0; c < cols; ++c) {
        delta[c] = b[0] * factor_a[c];
      }
      for (std::uint32_t k = 1; k < patch.rank; ++k) {
        const float* a = factor_a + std::size_t{k} * cols;
        for (std::size_t c = 0; c < cols; ++c) {
          const float product = b[k] * a[c];
          delta[c] = delta[c] + product;
        }
      }
      bool finite = true;
      for (std::size_t c = 0; c < cols; ++c) {
        const float scaled = delta[c] * multiplier;
        row[c] = row[c] + scaled;
        finite = finite && std::isfinite(row[c]);
      }
      if (!finite || !QuantizeRowQ8_0(
                         row.data(),
                         out + static_cast<std::size_t>(i) * row_bytes, cols)) {
#pragma omp critical
        failed = std::min(failed, i);
      }
    }
  }
  if (failed != rows) {
    if (error != nullptr) {
      *error = "LoRA merge of " + std::string(t.name) + " row " +
               std::to_string(first + static_cast<std::uint64_t>(failed)) +
               " is not finite in Q8_0";
    }
    return false;
  }
  return true;
}

bool QuantizeRowQ8_0(const float* x, std::uint8_t* out,
                     std::size_t count) noexcept {
  bool finite = true;
  for (std::size_t i = 0; i < count / kQ8Block; ++i) {
    const float* v = x + i * kQ8Block;
    float amax = 0.0F;
    for (std::size_t j = 0; j < kQ8Block; ++j) {
      amax = std::max(amax, std::fabs(v[j]));
    }
    const float d = amax / 127.0F;
    // A subnormal d has no finite reciprocal; its half scale is zero, so the
    // block encodes as zeros instead of converting infinities to codes.
    const float reciprocal = d != 0.0F ? 1.0F / d : 0.0F;
    const float id = std::isfinite(reciprocal) ? reciprocal : 0.0F;
    const std::uint16_t half = FloatToHalf(d);
    finite = finite && (half & 0x7C00U) != 0x7C00U;
    std::uint8_t* block = out + i * kQ8BlockBytes;
    std::memcpy(block, &half, sizeof(half));
    for (std::size_t j = 0; j < kQ8Block; ++j) {
      const auto code = static_cast<std::int8_t>(std::round(v[j] * id));
      std::memcpy(block + 2 + j, &code, 1);
    }
  }
  return finite;
}

std::uint16_t FloatToHalf(float value) noexcept {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000U);
  const std::uint32_t magnitude = bits & 0x7FFFFFFFU;
  if (magnitude > 0x7F800000U) {
    return sign | 0x7E00U;  // NaN
  }
  if (magnitude >= 0x47800000U) {
    return sign | 0x7C00U;  // 2^16 and above (and infinity) overflow
  }
  if (magnitude < 0x38800000U) {
    // Below 2^-14 the half is subnormal, in units of 2^-24. Scaling by a
    // power of two is exact, and nearbyint rounds to nearest even; 1024
    // carries into the smallest normal.
    const float units = std::fabs(value) * 16777216.0F;
    return sign | static_cast<std::uint16_t>(std::nearbyint(units));
  }
  std::uint32_t half =
      ((magnitude >> 23) - 112U) << 10 | (magnitude & 0x7FFFFFU) >> 13;
  const std::uint32_t rest = magnitude & 0x1FFFU;
  if (rest > 0x1000U || (rest == 0x1000U && (half & 1U) != 0)) {
    ++half;  // may carry into the exponent, up to infinity
  }
  return sign | static_cast<std::uint16_t>(half);
}

}  // namespace gufo::models::qwen::lora
