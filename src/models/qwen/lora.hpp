#ifndef GUFO_MODELS_QWEN_LORA_HPP_
#define GUFO_MODELS_QWEN_LORA_HPP_

#include <gufo/ext/options.hpp>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/families/qwen/lora/lora.hpp"
#include "src/models/qwen/state.hpp"

namespace gufo::models::qwen {

/// Names the tensors a Qwen3.8 27B adapter may patch, in its errors.
inline constexpr std::string_view kLoraExpectedTargets =
    "blk.N.ssm_out.weight, blk.N.attn_output.weight or blk.N.ffn_down.weight "
    "of a trunk layer";

/// Trunk matrices an adapter may patch: the GDN output, attention output and
/// FFN down projections of every trunk layer, under the name each was bound
/// from. Rows and columns come from the GGUF tensor's own dimensions and must
/// match the configuration, so a transposed tensor cannot pass on its element
/// count. The MTP block (blk.64) is not part of the trunk and is never a
/// target. No value, with `error`, when a bound tensor's shape disagrees.
[[nodiscard]] std::optional<std::vector<lora::LoraMatrix>> LoraTargets(
    const core::GgufReader& reader, const QwenModelWeights& weights,
    std::string* error);

/// Opens `selection.path` and validates it against the trunk `weights` bound
/// from `reader`, before any device memory is allocated. Null with `error`,
/// also when `LoraPatchesOwnTheirBytes` fails.
[[nodiscard]] std::unique_ptr<lora::LoraAdapter> OpenLoraAdapter(
    const core::GgufReader& reader, const QwenModelWeights& weights,
    const lora::LoraSelection& selection, std::string* error);

/// The 27B loader overwrites a patched tensor where it lies in its shard, so
/// bytes it shares with another tensor would change that tensor too. False,
/// with `error`, when a patch of `adapter` shares bytes with another tensor
/// of `reader`.
[[nodiscard]] bool LoraPatchesOwnTheirBytes(const core::GgufReader& reader,
                                            const lora::LoraAdapter& adapter,
                                            std::string* error);

/// The CPU generator cannot merge an adapter: false, with `error`, when the
/// options select one or are invalid.
[[nodiscard]] bool ValidateCpuLoraOptions(const ext::ParsedOptions& options,
                                          std::string* error);

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_LORA_HPP_
