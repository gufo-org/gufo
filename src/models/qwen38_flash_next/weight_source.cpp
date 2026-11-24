#include "src/models/qwen38_flash_next/weight_source.hpp"

#include <filesystem>
#include <optional>
#include <utility>

#include "src/families/qwen/text/tokenizer.hpp"
#include "src/models/qwen38_flash_next/config.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::ext::qwen::flash_next {
namespace {

/// The default source: a GGUF trunk (all its shards) and the optional MTP
/// sidecar, bound from their headers and uploaded from their mappings.
class GgufWeightSource final : public WeightSource {
public:
  [[nodiscard]] bool Probe(const std::string& model_path) const override {
    std::error_code ec;
    return std::filesystem::is_regular_file(model_path, ec);
  }

  bool Open(const std::string& model_path, const std::string& mtp_path,
            std::string* error) override {
    reader_ = core::GgufReader::OpenFile(model_path, error);
    if (!reader_)
      return false;
    if (!mtp_path.empty()) {
      mtp_reader_ = core::GgufReader::OpenFile(mtp_path, error);
      if (!mtp_reader_)
        return false;
    }
    return true;
  }

  bool BindTrunk(ModelWeights* weights, std::string* error) override {
    auto bound = ModelWeights::Bind(*reader_, error);
    if (!bound)
      return false;
    *weights = std::move(*bound);
    return true;
  }

  bool BindMtp(const Config& trunk, MtpWeights* mtp,
               std::string* error) override {
    if (!mtp_reader_)
      return false;
    auto bound = MtpWeights::Bind(*mtp_reader_, trunk, error);
    if (!bound)
      return false;
    *mtp = std::move(*bound);
    return true;
  }

  std::unique_ptr<tokenization::QwenTokenizer> Tokenizer(
      const Config& config, std::string* error) override {
    (void)config;
    return tokenization::QwenTokenizer::CreateFromGguf(*reader_, error);
  }

  std::unique_ptr<NgramTable> Ngram(const ModelWeights& weights,
                                    std::string* error) override {
    const auto& t = weights.ple_table;
    return NgramTable::Open(
        reader_->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
        t.rows, weights.config.ple_head_dim, t.type, error);
  }

  [[nodiscard]] std::span<const core::GgufMappedRegion> Regions(
      bool mtp) const override {
    if (mtp)
      return mtp_reader_ ? mtp_reader_->GetMappedRegions()
                         : std::span<const core::GgufMappedRegion>{};
    return reader_->GetMappedRegions();
  }

  [[nodiscard]] std::string ModelName() const override {
    return std::string(reader_->GetMetadataString("general.name")
                           .value_or("Qwen3.8-Flash-Next"));
  }

private:
  std::unique_ptr<core::GgufReader> reader_;
  std::unique_ptr<core::GgufReader> mtp_reader_;
};

}  // namespace

std::unique_ptr<WeightSource> CreateGgufWeightSource() {
  return std::make_unique<GgufWeightSource>();
}

}  // namespace gufo::ext::qwen::flash_next
