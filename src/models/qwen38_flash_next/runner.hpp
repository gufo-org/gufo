#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "gufo/ext/continuation.hpp"
#include "gufo/ext/text.hpp"

namespace gufo::ext::qwen::flash_next {
class Stages;
class WeightSource;
}  // namespace gufo::ext::qwen::flash_next

namespace gufo::models::qwen38_flash_next {
class Model;

inline constexpr std::uint32_t DecodeConcurrencyForSessions(
    std::size_t session_count) {
  return static_cast<std::uint32_t>(
      std::clamp<std::size_t>(session_count, 1, 8));
}

#if defined(ENGINE_ENABLE_HIP)
std::shared_ptr<ext::TextModelRunner> CreateQwenFlashNextTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

/// The serving runner over a non-GGUF artifact: `weights` (required) reads
/// `options.model_path` and `stages` (null: GGUF defaults) runs its tensors.
/// The native context comes from WeightSource::NativeContextLength (checked
/// against the bound Config), the MTP draft from WeightSource::MtpPath, and
/// the disk cache identities, unless the options name them, from Identity /
/// DraftIdentity: a draft block of the artifact itself (MtpPath returned the
/// model path) defaults to sha256(Identity() + ":mtp"), a separate GGUF
/// draft to its GGUF identity.
/// The disk cache identities of a runner and the scheme labelling each.
struct FlashNextCacheIdentities {
  std::string model;
  std::string draft;
  std::string_view model_scheme{ext::kGgufIdentityScheme};
  std::string_view draft_scheme{ext::kGgufIdentityScheme};
};

/// The non-GGUF runner's identities: fills the empty ones of `ids` from
/// `model`'s weight source (see below; `own_draft` when the MTP path is the
/// model path) and requires both to be SHA-256 hex.
[[nodiscard]] bool ResolveWeightSourceIdentities(const Model& model,
                                                 bool use_mtp, bool own_draft,
                                                 const std::string& mtp_path,
                                                 FlashNextCacheIdentities* ids,
                                                 std::string* error);

std::shared_ptr<ext::TextModelRunner> CreateQwenFlashNextTextRunner(
    const ext::TextRunnerOptions& options,
    std::unique_ptr<ext::qwen::flash_next::Stages> stages,
    std::unique_ptr<ext::qwen::flash_next::WeightSource> weights,
    std::string* error);

/// `artifact_scheme` / `draft_scheme` label the fingerprints in the disk
/// cache identity: GGUF identities by default, a weight source's own
/// identities as ext::qwen::flash_next::kWeightSourceIdentityScheme.
std::shared_ptr<ext::TextModelRunner> CreateQwenFlashNextTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    const ext::TextSpeculativeConfig& speculative_config = {},
    std::string artifact_fingerprint = {},
    std::string draft_artifact_fingerprint = {},
    std::string_view artifact_scheme = ext::kGgufIdentityScheme,
    std::string_view draft_scheme = ext::kGgufIdentityScheme);
#endif

}  // namespace gufo::models::qwen38_flash_next
