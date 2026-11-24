#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHT_SOURCE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHT_SOURCE_HPP_

// Where a Flash-Next model's weights, tokenizer and n-gram table come from.
// The GGUF artifact (trunk shards plus the MTP sidecar) is the default
// source (CreateGgufWeightSource); another source binds the same
// descriptors from its own files and uploads the tensors whose encoding
// only its Stages read (TensorRef::format_state). Experimental: no
// compatibility promise yet.
//
// Not yet self-contained for include/: it names Config, ModelWeights,
// MtpWeights and TensorRef (config.hpp, weights.hpp), NgramTable
// (ngram.hpp) and tokenization::QwenTokenizer, which must move or gain
// public counterparts with it. NgramTable only reads GGUF row formats, so
// another source cannot provide its own PLE table yet.

#include <cstddef>
#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "primitives.hpp"

namespace gufo::models::qwen38_flash_next {
struct Config;
struct ModelWeights;
struct MtpWeights;
struct TensorRef;
class NgramTable;
}  // namespace gufo::models::qwen38_flash_next

namespace gufo::tokenization {
class QwenTokenizer;
}  // namespace gufo::tokenization

namespace gufo::ext::qwen::flash_next {

/// The scheme labelling a weight source's own Identity() / DraftIdentity()
/// in disk cache identities; GGUF identities keep "gguf-sha256-v1".
inline constexpr std::string_view kWeightSourceIdentityScheme =
    "flash-next-weight-source-sha256-v1";

using ::gufo::models::qwen38_flash_next::Config;
using ::gufo::models::qwen38_flash_next::ModelWeights;
using ::gufo::models::qwen38_flash_next::MtpWeights;
using ::gufo::models::qwen38_flash_next::NgramTable;
using ::gufo::models::qwen38_flash_next::TensorRef;

/// The device upload as a source's own tensors see it. The model is walked
/// twice: a measuring walk sizes the weight heap, then the uploading walk
/// fills it; both must request the same allocations.
class UploadContext {
public:
  /// The measuring walk: Allocate only counts and nothing is copied; the
  /// returned Tensor is ignored.
  [[nodiscard]] virtual bool Measuring() const noexcept = 0;
  /// Weight heap memory, freed with the model.
  [[nodiscard]] virtual bool Allocate(void** pointer, std::size_t bytes) = 0;
  /// Queues a copy of `bytes` at `offset` of the uploaded block's region
  /// `region` (WeightSource::Regions) into device memory.
  [[nodiscard]] virtual bool CopyFromRegion(std::uint32_t region,
                                            std::uint64_t offset,
                                            std::size_t bytes,
                                            void* device) = 0;
  /// Waits for every queued copy.
  [[nodiscard]] virtual bool Finish() = 0;
  virtual void Fail(std::string_view message) = 0;

protected:
  ~UploadContext() = default;
};

/// What the upload knows about where a tensor is used.
struct UploadHint {
  /// A shared-expert projection, which runs beside the routed experts.
  bool beside_routed_experts{false};
};

class WeightSource {
public:
  virtual ~WeightSource() = default;

  /// Whether this source reads `model_path`; no file is kept open.
  [[nodiscard]] virtual bool Probe(const std::string& model_path) const = 0;
  /// Opens the trunk and, when `mtp_path` is not empty, the draft block.
  virtual bool Open(const std::string& model_path, const std::string& mtp_path,
                    std::string* error) = 0;
  /// Binds and validates the trunk descriptors. Payloads stay in the source
  /// (mapped regions, or host memory with TensorRef::shard == kHostShard).
  virtual bool BindTrunk(ModelWeights* weights, std::string* error) = 0;
  /// Binds the MTP draft block; called only after Open with an MTP path.
  virtual bool BindMtp(const Config& trunk, MtpWeights* mtp,
                       std::string* error) = 0;
  virtual std::unique_ptr<::gufo::tokenization::QwenTokenizer> Tokenizer(
      const Config& config, std::string* error) = 0;
  /// The PLE n-gram table (Config::ple_layer >= 0), read from disk per token.
  virtual std::unique_ptr<NgramTable> Ngram(const ModelWeights& weights,
                                            std::string* error) = 0;
  /// File regions TensorRef::shard indexes: the trunk's, or the draft
  /// block's with `mtp`.
  [[nodiscard]] virtual std::span<const ::gufo::core::GgufMappedRegion> Regions(
      bool mtp) const = 0;
  /// After each walk of the upload (the measuring one, then the uploading
  /// one, also when it failed), once every region read it queued has
  /// landed: a source may free its upload staging.
  virtual void UploadDone(bool measuring) { (void)measuring; }
  /// Uploads a tensor with format_state; GGUF tensors never reach it.
  virtual Tensor Upload(const TensorRef& tensor, const UploadHint& hint,
                        UploadContext& upload) {
    (void)tensor;
    (void)hint;
    upload.Fail("the weight source cannot upload this tensor format");
    return {};
  }
  /// After the upload: a row-addressable GGUF copy of the output head that
  /// draft tables copy rows from, when `output` has format_state. Empty
  /// keeps `output` itself. The source owns that memory.
  virtual bool DraftHead(const Tensor& output, Tensor* head,
                         std::string* error) {
    (void)output;
    (void)error;
    *head = {};
    return true;
  }
  /// Cache identity of the artifact as lowercase SHA-256 hex; empty lets
  /// callers hash the GGUF reader. The non-GGUF runner requires it when a
  /// disk cache is set.
  [[nodiscard]] virtual std::string Identity() const { return {}; }
  /// Cache identity (SHA-256 hex) of the bound MTP draft block, for the
  /// non-GGUF runner. Empty derives sha256(Identity() + ":mtp") when MtpPath
  /// returned the model path (the artifact's own block), and the GGUF
  /// identity of a separate GGUF draft model; otherwise it is required.
  [[nodiscard]] virtual std::string DraftIdentity() const { return {}; }
  /// Native context of the artifact at `model_path`, read before Open (the
  /// non-GGUF runner sizes the context and YaRN from it, then requires it to
  /// equal the bound Config::context_length); 0 if unknown.
  [[nodiscard]] virtual std::uint32_t NativeContextLength(
      const std::string& model_path) const {
    (void)model_path;
    return 0;
  }
  /// The MTP path to Open for a request of `requested` (the configured draft
  /// model, possibly empty): a checkpoint carrying its own draft block
  /// returns `model_path`.
  [[nodiscard]] virtual std::string MtpPath(
      const std::string& model_path, const std::string& requested) const {
    (void)model_path;
    return requested;
  }
  [[nodiscard]] virtual std::string ModelName() const = 0;
  /// Where the vision projector sidecar is searched from.
  [[nodiscard]] virtual std::string VisionPath(
      const std::string& model_path) const {
    return model_path;
  }
  /// Rows of a narrow prompt pass.
  [[nodiscard]] virtual std::uint32_t PreferredPrefillChunk() const {
    return 4096;
  }
};

/// The GGUF trunk shards plus the `shared-*` MTP sidecar.
[[nodiscard]] std::unique_ptr<WeightSource> CreateGgufWeightSource();

}  // namespace gufo::ext::qwen::flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHT_SOURCE_HPP_
