#ifndef GUFO_CORE_LORA_ADAPTER_HPP_
#define GUFO_CORE_LORA_ADAPTER_HPP_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// Model-independent LoRA adapter files: safetensors parsing, the published
// tensor layouts, alpha/scale rules and validation. A model describes which
// base tensors its module names patch with a `Schema`; merging the terms
// into its weights stays with the model's runtime.
namespace gufo::lora {

enum class Dtype { kBf16, kF16, kF32 };

/// Row-major view into a mapped adapter file.
struct Tensor {
  const void* data{nullptr};
  Dtype dtype{Dtype::kF32};
  std::vector<int> shape;
};

/// Elements [first, first + count) as FP32 (exact for all three dtypes).
std::vector<float> Floats(const Tensor& tensor, std::size_t first,
                          std::size_t count);

/// One update of rows [base_row, base_row + rows) of one base tensor, from
/// rows [first_row, first_row + rows) of the module's up/diff/magnitude.
struct Patch {
  // Base tensor, e.g. "transformer.transformer_blocks.0.attn.to_q.weight".
  std::string target;
  int rows{0}, cols{0};  // patched rows; 1-D targets have cols == 1
  int base_row{0};       // first patched row of the target
  int rank{0};           // 0 for a diff
  int first_row{0};      // first row used of up/diff/magnitude
  float scale{1};        // alpha / rank; strength is applied by the caller
  Tensor down, up, diff, magnitude;  // absent tensors have data == nullptr
};

struct Term {
  const Patch* patch{nullptr};
  float strength{1};
};

/// Rows of a module's update that land on one base tensor. Consecutive
/// slices of a module take consecutive rows of its up/diff/magnitude.
struct Slice {
  std::string target;
  int base_row{0};
  int rows{0};  // 0: every row of the target from base_row
};

/// How a model's adapter module names map onto its base tensors.
struct Schema {
  /// Base tensor shape by name, or empty when the tensor does not exist.
  std::function<std::vector<int>(std::string_view)> base;
  /// Canonical module name (no prefix, no suffix) to the rows it patches.
  /// Kohya names (`lora_unet_` + dots as underscores) derive from the keys.
  std::map<std::string, std::vector<Slice>, std::less<>> modules;
  /// Key prefixes stripped before lookup; the first match is removed.
  std::vector<std::string> prefixes;
  /// Reason for a module that is not in `modules`, given the name after
  /// prefix removal (kohya names keep `lora_unet_`).
  std::function<std::string(std::string_view module)> unknown;
};

/// A validated adapter file (LoRA, DoRA, diffs) against one schema.
class Adapter {
public:
  Adapter(const std::filesystem::path& path, const Schema& schema);
  const std::vector<Patch>& patches() const { return patches_; }
  /// Layout name with features, e.g. "kohya+alpha+gate_up".
  const std::string& format() const { return format_; }
  /// Smallest and largest pair rank; 0 if the file only has diffs.
  int min_rank() const;
  int max_rank() const;

private:
  std::shared_ptr<void> mapping_;
  std::vector<Patch> patches_;
  std::string format_;
};

}  // namespace gufo::lora
#endif
