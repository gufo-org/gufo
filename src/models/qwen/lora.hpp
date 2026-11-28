#ifndef GUFO_MODELS_QWEN_LORA_HPP_
#define GUFO_MODELS_QWEN_LORA_HPP_

#include <cstddef>
#include <cstdint>
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

/// A patchable trunk matrix and where it sits: its layer, and whether it is
/// the layer's FFN down projection rather than its attention or GDN output.
struct LoraTarget {
  lora::LoraMatrix matrix;
  std::uint32_t layer{0};
  bool ffn_down{false};
};

/// `LoraTargets` with each matrix's layer and role, from the names the
/// tensors were bound under.
[[nodiscard]] std::optional<std::vector<LoraTarget>> LoraTargetsByLayer(
    const core::GgufReader& reader, const QwenModelWeights& weights,
    std::string* error);

/// Opens `selection.path` and validates it against the trunk `weights` bound
/// from `reader`, before any device memory is allocated. Targets may be in
/// any format. Null with `error`, also when the adapter merges in place and
/// `LoraPatchesOwnTheirBytes` fails.
[[nodiscard]] std::unique_ptr<lora::LoraAdapter> OpenLoraAdapter(
    const core::GgufReader& reader, const QwenModelWeights& weights,
    const lora::LoraSelection& selection, std::string* error);

/// True when every tensor `adapter` patches is Q8_0, so the loader merges it
/// into the weights. Otherwise the stock weights stay untouched and the
/// update is added beside each patched projection at inference (the side
/// path), Q8_0 targets included.
[[nodiscard]] bool LoraMergesInPlace(const lora::LoraAdapter& adapter);

/// Names the side path's arithmetic in the cache identity: stock weights,
/// and multiplier * B * (A * x) added in float32 to the projection's output.
inline constexpr std::string_view kLoraSidePathAlgorithm = "low-rank-fp32-v1";
/// Largest adapter rank the side path accepts.
inline constexpr std::uint32_t kLoraSidePathMaxRank = 64;

/// The `adapters=` value of the cache identity for a loaded `adapter`: its
/// merge algorithm when it merges in place, the side path's otherwise.
[[nodiscard]] std::string LoraAdapterIdentity(const lora::LoraAdapter& adapter);

/// One patch of the side path: the layer projection it sits beside, its
/// dimensions and multiplier, and where its factors lie in
/// `LoraSidePath::factors`.
struct LoraSidePathSlot {
  std::uint32_t layer{0};
  bool ffn_down{false};  ///< Otherwise beside attn_output or ssm_out.
  std::uint32_t in{0};
  std::uint32_t out{0};
  std::uint32_t rank{0};
  float multiplier{0.0F};
  std::size_t a{0};  ///< Float offset of A, [rank][in].
  std::size_t b{0};  ///< Float offset of B, [out][rank].
};

/// Every side-path factor in one float32 buffer, each A and B starting at a
/// multiple of 16 floats (64 bytes), ready for a single upload.
struct LoraSidePath {
  std::vector<float> factors;
  std::vector<LoraSidePathSlot> slots;
};

/// Lays out `adapter` for the side path over `weights` as bound from
/// `reader`, placing each patch by its tensor name. Null, with `error` naming
/// the tensor, when a patch has a rank above `kLoraSidePathMaxRank`, an input
/// width not divisible by 4, a non-finite multiplier, or no bound projection.
[[nodiscard]] std::optional<LoraSidePath> PlanLoraSidePath(
    const core::GgufReader& reader, const lora::LoraAdapter& adapter,
    const QwenModelWeights& weights, std::string* error);

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
