#ifndef GUFO_MODELS_QWEN_HRX_QWEN_HRX_MANIFEST_HPP_
#define GUFO_MODELS_QWEN_HRX_QWEN_HRX_MANIFEST_HPP_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen/hrx/qwen_hrx_contract.hpp"

namespace gufo::hrx {

struct HrxArtifactManifestEntry {
  std::string name;
  std::string filename;
  std::string export_name;
  std::uint32_t binding_count{0};
  std::vector<std::string> binding_order;
  std::vector<std::string> scalar_constants;
  std::string tensor_encoding;
  std::uint32_t workgroup_size[3]{1, 1, 1};
  std::uint32_t subgroup_size{32};
  std::string sha256;
  bool optional{false};
};

class HrxArtifactManifest {
public:
  static std::unique_ptr<HrxArtifactManifest> LoadFromFile(
      const std::string& manifest_path, std::string* error_msg = nullptr);
  static std::unique_ptr<HrxArtifactManifest> LoadFromDirectory(
      const std::string& kernels_dir, std::string* error_msg = nullptr);
  static std::unique_ptr<HrxArtifactManifest> CreateBuiltin();

  [[nodiscard]] bool ValidateDirectory(const std::string& kernels_dir,
                                       const QwenHrxArtifactContract& contract,
                                       std::string* error_msg = nullptr) const;

  [[nodiscard]] const std::string& SchemaVersion() const noexcept {
    return schema_version_;
  }
  [[nodiscard]] const std::string& ModelKind() const noexcept {
    return model_kind_;
  }
  [[nodiscard]] const std::string& Target() const noexcept { return target_; }
  [[nodiscard]] std::uint32_t WaveSize() const noexcept { return wave_size_; }
  [[nodiscard]] const std::string& HrxAbiRevision() const noexcept {
    return hrx_abi_revision_;
  }
  [[nodiscard]] const std::vector<HrxArtifactManifestEntry>& Entries()
      const noexcept {
    return entries_;
  }
  [[nodiscard]] const HrxArtifactManifestEntry* FindEntry(
      std::string_view name) const noexcept;
  [[nodiscard]] const HrxArtifactManifestEntry* FindEntryByFilename(
      std::string_view filename) const noexcept;

  void SetSchemaVersion(std::string version) {
    schema_version_ = std::move(version);
  }
  void SetModelKind(std::string kind) { model_kind_ = std::move(kind); }
  void SetTarget(std::string target) { target_ = std::move(target); }
  void SetWaveSize(std::uint32_t wave_size) { wave_size_ = wave_size; }
  void SetHrxAbiRevision(std::string rev) {
    hrx_abi_revision_ = std::move(rev);
  }
  void AddEntry(HrxArtifactManifestEntry entry) {
    entries_.push_back(std::move(entry));
  }

private:
  std::string schema_version_{"1.0.0"};
  std::string model_kind_{"qwen3.8-27b"};
  std::string target_{"gfx1151"};
  std::uint32_t wave_size_{32};
  std::string hrx_abi_revision_{"hrx-loom-v1"};
  std::vector<HrxArtifactManifestEntry> entries_;
};

}  // namespace gufo::hrx

#endif  // GUFO_MODELS_QWEN_HRX_QWEN_HRX_MANIFEST_HPP_
