#include "src/models/qwen/hrx/qwen_hrx_manifest.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_set>

#include "src/cli/serve/json.hpp"
#include "src/models/minimax_h3/sha256.hpp"

namespace gufo::hrx {

namespace {

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
}

}  // namespace

const HrxArtifactManifestEntry* HrxArtifactManifest::FindEntry(
    std::string_view name) const noexcept {
  for (const auto& entry : entries_) {
    if (entry.name == name) {
      return &entry;
    }
  }
  return nullptr;
}

const HrxArtifactManifestEntry* HrxArtifactManifest::FindEntryByFilename(
    std::string_view filename) const noexcept {
  for (const auto& entry : entries_) {
    if (entry.filename == filename) {
      return &entry;
    }
  }
  return nullptr;
}

std::unique_ptr<HrxArtifactManifest> HrxArtifactManifest::LoadFromFile(
    const std::string& manifest_path, std::string* error_msg) {
  if (!std::filesystem::exists(manifest_path)) {
    SetError(error_msg, "HRX manifest file does not exist: " + manifest_path);
    return nullptr;
  }
  std::ifstream file(manifest_path);
  if (!file.is_open()) {
    SetError(error_msg, "failed to open HRX manifest file: " + manifest_path);
    return nullptr;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  const std::string content = buffer.str();

  try {
    const auto root = server::json::parse(content);
    if (!root.is_object()) {
      SetError(error_msg, "HRX manifest root is not a JSON object");
      return nullptr;
    }

    auto manifest = std::make_unique<HrxArtifactManifest>();
    if (auto* sv = root.find("schema_version"); sv && sv->is_string()) {
      manifest->schema_version_ = sv->str();
    }
    if (auto* mk = root.find("model_kind"); mk && mk->is_string()) {
      manifest->model_kind_ = mk->str();
    }
    if (auto* tgt = root.find("target"); tgt && tgt->is_string()) {
      manifest->target_ = tgt->str();
    }
    if (auto* ws = root.find("wave_size"); ws && ws->is_number()) {
      manifest->wave_size_ = static_cast<std::uint32_t>(ws->as_size());
    }
    if (auto* abi = root.find("hrx_abi_revision"); abi && abi->is_string()) {
      manifest->hrx_abi_revision_ = abi->str();
    }

    if (auto* entries = root.find("entries"); entries && entries->is_array()) {
      for (const auto& item : entries->items()) {
        if (!item.is_object())
          continue;
        HrxArtifactManifestEntry entry;
        if (auto* name = item.find("name"); name && name->is_string()) {
          entry.name = name->str();
        }
        if (auto* fn = item.find("filename"); fn && fn->is_string()) {
          entry.filename = fn->str();
        }
        if (auto* exp = item.find("export_name"); exp && exp->is_string()) {
          entry.export_name = exp->str();
        }
        if (auto* bc = item.find("binding_count"); bc && bc->is_number()) {
          entry.binding_count = static_cast<std::uint32_t>(bc->as_size());
        }
        if (auto* bo = item.find("binding_order"); bo && bo->is_array()) {
          for (const auto& b : bo->items()) {
            if (b.is_string())
              entry.binding_order.push_back(b.str());
          }
        }
        if (auto* sc = item.find("scalar_constants"); sc && sc->is_array()) {
          for (const auto& c : sc->items()) {
            if (c.is_string())
              entry.scalar_constants.push_back(c.str());
          }
        }
        if (auto* te = item.find("tensor_encoding"); te && te->is_string()) {
          entry.tensor_encoding = te->str();
        }
        if (auto* wg = item.find("workgroup_size"); wg && wg->is_array()) {
          const auto& arr = wg->items();
          if (arr.size() >= 3) {
            entry.workgroup_size[0] =
                static_cast<std::uint32_t>(arr[0].as_size());
            entry.workgroup_size[1] =
                static_cast<std::uint32_t>(arr[1].as_size());
            entry.workgroup_size[2] =
                static_cast<std::uint32_t>(arr[2].as_size());
          }
        }
        if (auto* sg = item.find("subgroup_size"); sg && sg->is_number()) {
          entry.subgroup_size = static_cast<std::uint32_t>(sg->as_size());
        }
        if (auto* sha = item.find("sha256"); sha && sha->is_string()) {
          entry.sha256 = sha->str();
        }
        if (auto* opt = item.find("optional"); opt && opt->is_bool()) {
          entry.optional = opt->as_bool();
        }
        manifest->entries_.push_back(std::move(entry));
      }
    }
    if (error_msg != nullptr) {
      error_msg->clear();
    }
    return manifest;
  } catch (const std::exception& ex) {
    SetError(error_msg,
             std::string("failed to parse HRX manifest JSON: ") + ex.what());
    return nullptr;
  }
}

std::unique_ptr<HrxArtifactManifest> HrxArtifactManifest::LoadFromDirectory(
    const std::string& kernels_dir, std::string* error_msg) {
  const auto manifest_path =
      (std::filesystem::path(kernels_dir) / "hrx_manifest.json").string();
  if (std::filesystem::exists(manifest_path)) {
    return LoadFromFile(manifest_path, error_msg);
  }
  // Fallback to built-in default schema if standalone manifest json file is not
  // yet deployed
  auto manifest = CreateBuiltin();
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  return manifest;
}

std::unique_ptr<HrxArtifactManifest> HrxArtifactManifest::CreateBuiltin() {
  auto manifest = std::make_unique<HrxArtifactManifest>();
  manifest->schema_version_ = "1.0.0";
  manifest->model_kind_ = "qwen3.8-27b";
  manifest->target_ = "gfx1151";
  manifest->wave_size_ = 32;
  manifest->hrx_abi_revision_ = "hrx-loom-v1";

  const struct BuiltinSpec {
    const char* name;
    const char* filename;
    const char* export_name;
    const char* tensor_encoding;
    std::uint32_t binding_count;
    std::uint32_t sg;
    std::uint32_t wg[3];
    bool optional;
  } builtin_specs[] = {
      {"qwen_swiglu",
       "qwen_fused_swiglu_bf16.fb",
       "qwen_fused_swiglu_bf16",
       "bf16",
       4,
       32,
       {128, 1, 1},
       false},
      {"qwen_rmsnorm_qkv",
       "qwen_fused_rmsnorm_qkv_bf16.fb",
       "qwen_fused_rmsnorm_qkv_bf16",
       "bf16",
       4,
       32,
       {128, 1, 1},
       false},
      {"qwen_rope_kv",
       "qwen_fused_rope_kv_cache_bf16.fb",
       "qwen_fused_rope_kv_cache_bf16",
       "bf16",
       7,
       32,
       {128, 1, 1},
       false},
      {"qwen_down_residual",
       "qwen_fused_down_residual_bf16.fb",
       "qwen_fused_down_residual_bf16",
       "bf16",
       4,
       32,
       {128, 1, 1},
       false},
      {"qwen_rmsnorm",
       "qwen_rmsnorm_f32.fb",
       "qwen_rmsnorm_f32",
       "f32",
       3,
       32,
       {128, 1, 1},
       false},
      {"qwen_residual_add",
       "qwen_residual_add_f32.fb",
       "qwen_residual_add_f32",
       "f32",
       3,
       32,
       {256, 1, 1},
       false},
      {"qwen_swiglu_pointwise",
       "qwen_swiglu_pointwise_f32.fb",
       "qwen_swiglu_pointwise_f32",
       "f32",
       3,
       32,
       {256, 1, 1},
       false},
      {"qwen_split_q_gate",
       "qwen_split_q_gate_f32.fb",
       "qwen_split_q_gate_f32",
       "f32",
       3,
       32,
       {192, 1, 1},
       false},
      {"qwen_copy",
       "qwen_copy_f32.fb",
       "qwen_copy_f32",
       "f32",
       2,
       32,
       {256, 1, 1},
       false},
      {"qwen_q8_decode_oracle",
       "qwen_q8_0_decode_oracle.fb",
       "qwen_q8_0_decode_oracle",
       "q8_0",
       3,
       32,
       {160, 1, 1},
       false},
      {"qwen_q8_embedding",
       "qwen_q8_0_embedding_k5120.fb",
       "qwen_q8_0_embedding_k5120",
       "q8_0",
       2,
       32,
       {32, 1, 1},
       false},
      {"qwen_q8_gemv_k5120",
       "qwen_q8_0_gemv_k5120.fb",
       "qwen_q8_0_gemv_k5120",
       "q8_0",
       3,
       32,
       {160, 1, 1},
       false},
      {"qwen_q8_gemv_k6144",
       "qwen_q8_0_gemv_k6144.fb",
       "qwen_q8_0_gemv_k6144",
       "q8_0",
       3,
       32,
       {192, 1, 1},
       false},
      {"qwen_q8_gemv_k17408",
       "qwen_q8_0_gemv_k17408.fb",
       "qwen_q8_0_gemv_k17408",
       "q8_0",
       3,
       32,
       {544, 1, 1},
       false},
      {"qwen_q8_gemv_k17408_wg256",
       "qwen_q8_0_gemv_k17408_wg256.fb",
       "qwen_q8_0_gemv_k17408_wg256",
       "q8_0",
       3,
       32,
       {256, 1, 1},
       true},
      {"qwen_q8_vocab_gemv_k5120",
       "qwen_q8_0_vocab_gemv_k5120.fb",
       "qwen_q8_0_vocab_gemv_k5120",
       "q8_0",
       3,
       32,
       {160, 1, 1},
       false},
      {"qwen_per_head_rmsnorm",
       "qwen_per_head_rmsnorm_f32.fb",
       "qwen_per_head_rmsnorm_f32",
       "f32",
       3,
       32,
       {128, 1, 1},
       false},
      {"qwen_attention_decode",
       "qwen_attention_decode_f32.fb",
       "qwen_attention_decode_f32",
       "f32",
       5,
       32,
       {192, 1, 1},
       false},
      {"qwen_ssm_conv",
       "qwen_ssm_conv_silu_f32.fb",
       "qwen_ssm_conv_silu_f32",
       "f32",
       4,
       32,
       {256, 1, 1},
       false},
      {"qwen_deltanet_prepare",
       "qwen_deltanet_prepare_f32.fb",
       "qwen_deltanet_prepare_f32",
       "f32",
       4,
       32,
       {48, 1, 1},
       false},
      {"qwen_deltanet_recurrence",
       "qwen_deltanet_recurrence_f32.fb",
       "qwen_deltanet_recurrence_f32",
       "f32",
       7,
       32,
       {128, 1, 1},
       false},
      {"qwen_argmax",
       "qwen_argmax_f32.fb",
       "qwen_argmax_f32",
       "f32",
       2,
       32,
       {1, 1, 1},
       false},
  };

  for (const auto& spec : builtin_specs) {
    HrxArtifactManifestEntry entry;
    entry.name = spec.name;
    entry.filename = spec.filename;
    entry.export_name = spec.export_name;
    entry.binding_count = spec.binding_count;
    entry.tensor_encoding = spec.tensor_encoding;
    entry.workgroup_size[0] = spec.wg[0];
    entry.workgroup_size[1] = spec.wg[1];
    entry.workgroup_size[2] = spec.wg[2];
    entry.subgroup_size = spec.sg;
    entry.optional = spec.optional;
    manifest->entries_.push_back(std::move(entry));
  }
  return manifest;
}

bool HrxArtifactManifest::ValidateDirectory(
    const std::string& kernels_dir, const QwenHrxArtifactContract& contract,
    std::string* error_msg) const {
  if (schema_version_ != "1.0.0") {
    SetError(error_msg,
             "manifest schema version mismatch: expected 1.0.0, got " +
                 schema_version_);
    return false;
  }
  if (model_kind_ != "qwen3.8-27b") {
    SetError(error_msg,
             "manifest model kind mismatch: expected qwen3.8-27b, got " +
                 model_kind_);
    return false;
  }
  if (target_ != "gfx1151") {
    SetError(error_msg,
             "manifest target mismatch: expected gfx1151, got " + target_);
    return false;
  }
  if (wave_size_ != 32) {
    SetError(error_msg, "manifest wave size mismatch: expected 32, got " +
                            std::to_string(wave_size_));
    return false;
  }
  if (hrx_abi_revision_ != "hrx-loom-v1") {
    SetError(error_msg,
             "manifest ABI revision mismatch: expected hrx-loom-v1, got " +
                 hrx_abi_revision_);
    return false;
  }

  // Contract verification
  if (contract.HiddenSize() != 5120 || contract.FfnSize() != 17408 ||
      contract.VocabSize() != 248320 || contract.NumLayers() != 64) {
    SetError(
        error_msg,
        "model contract dimensions mismatch with Qwen3.8-27B artifact ABI");
    return false;
  }

  std::unordered_set<std::string> seen_names;
  std::unordered_set<std::string> seen_filenames;

  for (const auto& entry : entries_) {
    if (entry.name.empty() || entry.filename.empty() ||
        entry.export_name.empty()) {
      SetError(error_msg,
               "manifest entry contains empty name, filename, or export_name");
      return false;
    }
    if (entry.binding_count == 0) {
      SetError(error_msg,
               "artifact " + entry.filename + " has invalid binding_count 0");
      return false;
    }
    if (!seen_names.insert(entry.name).second) {
      SetError(error_msg,
               "manifest contains duplicate entry name: " + entry.name);
      return false;
    }
    if (!seen_filenames.insert(entry.filename).second) {
      SetError(error_msg,
               "manifest contains duplicate filename: " + entry.filename);
      return false;
    }

    const auto fb_path = std::filesystem::path(kernels_dir) / entry.filename;
    const bool exists = std::filesystem::exists(fb_path);

    if (!exists) {
      if (!entry.optional) {
        SetError(error_msg, "artifact " + entry.filename +
                                " missing: required artifact not found in " +
                                kernels_dir);
        return false;
      }
      continue;
    }

    // Verify SHA-256 hash if present in manifest
    if (!entry.sha256.empty()) {
      const std::string actual_sha = minimax_h3::Sha256File(fb_path);
      if (actual_sha != entry.sha256) {
        SetError(error_msg, "artifact " + entry.filename +
                                " sha256 mismatch: expected " + entry.sha256 +
                                ", got " + actual_sha);
        return false;
      }
    }
  }

  if (error_msg != nullptr) {
    error_msg->clear();
  }
  return true;
}

}  // namespace gufo::hrx
