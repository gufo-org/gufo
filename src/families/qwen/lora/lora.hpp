#ifndef GUFO_FAMILIES_QWEN_LORA_LORA_HPP_
#define GUFO_FAMILIES_QWEN_LORA_LORA_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/options.hpp>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::qwen::lora {

/// Names the merge arithmetic in the cache identity: rows dequantized from
/// Q8_0, the update added in float32 and the result encoded as Q8_0 again.
inline constexpr std::string_view kLoraMergeAlgorithm = "q8-requant-v1";

/// The adapter `--lora` and `--lora-scale` select; an empty path selects none.
struct LoraSelection {
  std::string path;
  float scale{1.0F};
};

/// Reads `--lora` and `--lora-scale` from a provider's parsed options.
/// Rejects a scale without an adapter and a scale that is not a finite
/// float32.
[[nodiscard]] bool ResolveLoraOptions(const ext::ParsedOptions& options,
                                      LoraSelection* selection,
                                      std::string* error);

/// A model matrix an adapter may patch, in the model's GGUF mapping: `rows`
/// rows of `cols` elements (GGUF dimensions [cols, rows]).
struct LoraMatrix {
  std::string_view name;  ///< GGUF tensor name; outlives the adapter.
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t rows{0};
  std::uint64_t cols{0};

  /// Encoded bytes of one row in this format, or 0 when unsupported.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows;
  }
};

/// One validated low-rank update of a Q8_0 model matrix:
/// W[row][col] += (sum_k B[row][k] * A[k][col]) * multiplier.
/// The factors are widened to float32 once, whatever the adapter stores.
struct LoraPatch {
  LoraMatrix target;  ///< The stock tensor in the model mapping.
  std::uint32_t rank{0};
  std::vector<float> a;  ///< [rank][target.cols]
  std::vector<float> b;  ///< [target.rows][rank]
};

/// A llama.cpp GGUF LoRA adapter validated against the model matrices it
/// patches. The merged weights keep the stock format, shape and kernels.
class LoraAdapter {
public:
  /// Opens and validates `path`. `architecture` is the model's
  /// `general.architecture`; `targets` are the matrices the model allows an
  /// adapter to patch, and `expected_targets` names them in the error for
  /// any other tensor ("blk.N.ssm_out.weight ... of a trunk layer").
  [[nodiscard]] static std::unique_ptr<LoraAdapter> Open(
      const std::filesystem::path& path, float scale,
      std::string_view architecture, std::span<const LoraMatrix> targets,
      std::string_view expected_targets, std::string* error);
  /// Validates an already opened adapter; `label` names it in errors.
  [[nodiscard]] static std::unique_ptr<LoraAdapter> Open(
      std::unique_ptr<core::GgufReader> reader, std::string label, float scale,
      std::string_view architecture, std::span<const LoraMatrix> targets,
      std::string_view expected_targets, std::string* error);

  /// The patch for the model tensor named `name`, or null.
  [[nodiscard]] const LoraPatch* Find(std::string_view name) const;
  /// Every patch, by model tensor name.
  [[nodiscard]] const std::map<std::string, LoraPatch, std::less<>>& patches()
      const noexcept {
    return patches_;
  }
  [[nodiscard]] std::size_t size() const noexcept { return patches_.size(); }
  [[nodiscard]] const std::string& label() const noexcept { return label_; }
  /// SHA-256 of the adapter file bytes that were validated.
  [[nodiscard]] const std::string& sha256() const noexcept { return sha256_; }
  [[nodiscard]] float scale() const noexcept { return scale_; }
  /// scale * alpha / rank, per patch.
  [[nodiscard]] float Multiplier(const LoraPatch& patch) const noexcept;
  /// Value of the cache identity's `adapters=` line: merge algorithm,
  /// adapter digest and the float32 bits of the effective scale.
  [[nodiscard]] std::string CacheIdentity() const;

  /// Writes rows [first, first + count) of the patched tensor, Q8_0 encoded,
  /// to `out` (count * target.RowBytes() bytes). False when a merged value
  /// or a block scale is not finite.
  [[nodiscard]] bool MergeRows(const LoraPatch& patch, std::uint64_t first,
                               std::uint64_t count, std::uint8_t* out,
                               std::string* error) const;

private:
  LoraAdapter() = default;

  std::string label_;
  std::string sha256_;
  float scale_{1.0F};
  float alpha_{1.0F};
  std::map<std::string, LoraPatch, std::less<>> patches_;
};

/// Encodes `count` floats (a multiple of 32) as Q8_0 exactly as llama.cpp's
/// quantize_row_q8_0_ref and gguf-py: d = max|x| / 127, codes
/// roundf(x * (1 / d)), d stored as IEEE half rounded to nearest even.
/// False when a block scale does not fit a finite half.
[[nodiscard]] bool QuantizeRowQ8_0(const float* x, std::uint8_t* out,
                                   std::size_t count) noexcept;

/// IEEE binary16 bits of `value`, rounded to nearest even.
[[nodiscard]] std::uint16_t FloatToHalf(float value) noexcept;

}  // namespace gufo::models::qwen::lora

#endif  // GUFO_FAMILIES_QWEN_LORA_LORA_HPP_
