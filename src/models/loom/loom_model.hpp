// Loom package probing and the shared model handle (libloom_text_embed).
#ifndef GUFO_LOOM_MODEL_HPP_
#define GUFO_LOOM_MODEL_HPP_

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gufo/ext/generated/loom_constants.hpp>
#include <gufo/ext/host.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/sha256.hpp>
#include <gufo/ext/text.hpp>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "loom_api.hpp"
#include "loom_package.hpp"

namespace gufo::loom {

namespace fs = std::filesystem;
using ext::TextRunnerToken;

inline constexpr std::uint64_t kExtentSlack = 1024;
// Retained row checkpoints, and host blocks naming them in Gufo's cache.
inline constexpr std::size_t kCheckpoints = 64;
inline constexpr std::size_t kMarkerBytes = 16;
// Marker blocks plus the cache's per-slot metadata.
inline constexpr std::size_t kHostCacheBytes = std::size_t{8} << 20;
inline constexpr std::size_t kTextBuffer = 1 << 16;

// The Loom SDK release of the linked library (a library built from an older
// copy of this repository's loom-embed does not say: no check).
#ifdef LOOM_TEXT_EMBED_HRX_LOOM
inline constexpr std::string_view kRuntimeSdk = LOOM_TEXT_EMBED_HRX_LOOM;
#else
inline constexpr std::string_view kRuntimeSdk = {};
#endif

// ---------------------------------------------------------------- probe ---

inline std::optional<fs::path> ManifestPath(std::string_view model_root) {
  std::error_code error;
  fs::path path(model_root);
  if (fs::is_directory(path, error))
    path /= "model.json";
  if (path.filename() != "model.json" || !fs::is_regular_file(path, error))
    return std::nullopt;
  return path;
}

// A Loom model card with programs and a service of kind "text".
inline bool ProbeLoomText(std::string_view model_root) {
  const auto manifest = ManifestPath(model_root);
  if (!manifest)
    return false;
  std::error_code error;
  if (fs::file_size(*manifest, error) > (std::size_t{4} << 20) || error)
    return false;
  std::ifstream in(*manifest, std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(in)), {});
  try {
    const auto card = json::parse(text);
    const auto* services = card.find("services");
    if (!card.contains("schema_version") || !card.contains("programs") ||
        services == nullptr || !services->is_object())
      return false;
    for (const auto& [name, service] : services->members()) {
      (void)name;
      if (service.is_object() && service.member_str("kind") == "text")
        return true;
    }
  } catch (const std::exception&) {
  }
  return false;
}

// ---------------------------------------------------------------- model ---

// The option that gave the draft model, for errors.
inline std::string DraftOption(const ext::TextSpeculativeConfig& speculative) {
  switch (speculative.backend) {
    case ext::TextSpeculativeBackend::kDFlash:
      return "--dflash-model";
    case ext::TextSpeculativeBackend::kDSpark:
      return "--dspark-model";
    default:
      return "--mtp-model";
  }
}

// The --speculative value that selected the backend, for errors.
inline std::string BackendName(const ext::TextSpeculativeConfig& speculative) {
  switch (speculative.backend) {
    case ext::TextSpeculativeBackend::kDFlash:
      return "dflash2";
    case ext::TextSpeculativeBackend::kDSpark:
      return "dspark";
    default:
      return "mtp";
  }
}

// One opened package; every Loom call is serialized on its mutex (recursive:
// releasing a cache entry can release a checkpoint inside a runner call).
class LoomModel {
public:
  // Gufo's own options size and complete the package: --sessions rows, each
  // able to reach --context (the package's context by default) at once, and
  // --mmproj or the draft model in place of the card's vision or draft file.
  // Loom opens its default device, the first AMD GPU.
  explicit LoomModel(const ext::TextRunnerOptions& options) {
    const std::string& path = options.model_path;
    const bool checkpoint_images = !options.disk_cache.directory.empty();
    std::map<std::string, package::Override> overrides;
    if (!options.vision_model_path.empty())
      overrides.emplace(
          "vision", package::Override{options.vision_model_path, "--mmproj"});
    if (!options.speculative.draft_model_path.empty())
      overrides.emplace("draft",
                        package::Override{options.speculative.draft_model_path,
                                          DraftOption(options.speculative)});
    // The preset follows the GPU Loom opens by default, as its driver reports
    // it on Linux and Windows alike.
    package::Gpu gpu;
    loom_text_embed_gpu_t found{};
    char gpu_error[512] = {};
    if (loom_text_embed_query_gpu(&found, gpu_error, sizeof(gpu_error)) == 0)
      gpu = {.target = found.target, .memory_bytes = found.memory_bytes};
    else
      Warn(std::string("cannot read the GPU from the Loom driver (") +
           gpu_error + ")");
    package_ = package::Resolve(path, overrides, gpu, kRuntimeSdk);
    const std::string preset = package_.preset;
    if (!package_.note.empty())
      std::fprintf(stderr, "loom: %s\n", package_.note.c_str());
    if (!package_.warning.empty())
      Warn(package_.warning);
    if (!package_.overlay.empty())
      std::fprintf(stderr,
                   "loom: parameter files found by origin in the model "
                   "directory\n");
    loom_text_embed_options_ex_t embed_options{};
    embed_options.struct_size = sizeof(embed_options);
    embed_options.base = {
        .model_path = package_.load_path.c_str(),
        .preset = preset.c_str(),
        .device_uri = nullptr,
        // Gufo sizes the pool in tokens (row_context), not bytes.
        .memory_bytes = 0,
        .rows = static_cast<std::uint32_t>(
            std::max<std::size_t>(options.session_count, 1)),
        .checkpoints = kCheckpoints,
    };
    // Every row's full context, committed at load: Gufo admits requests per
    // session, so the pool never refuses one it admitted.
    embed_options.row_context =
        options.max_context ? options.max_context : UINT32_MAX;
    // The logits capability whenever the package declares it.
    embed_options.logits = DeclaresLogits(path);
    // A disk tier stores checkpoint images, which need elastic state.
    embed_options.elastic = checkpoint_images;
    char error[2048] = {};
    if (loom_text_embed_open_ex(&embed_options, &embed_, error,
                                sizeof(error))) {
      if (embed_options.logits) {
        // A package whose state cannot return rows still serves, as before.
        std::fprintf(stderr,
                     "loom: logits unavailable (%s); sampling stays on the "
                     "device\n",
                     error);
        embed_options.logits = 0;
        error[0] = 0;
        if (loom_text_embed_open_ex(&embed_options, &embed_, error,
                                    sizeof(error)))
          throw std::runtime_error(std::string("loom: ") + error);
      } else {
        throw std::runtime_error(std::string("loom: ") + error);
      }
    }
    loom_text_embed_info(embed_, &info_);
    // Loom drafts with the package's drafter: speculation without one would
    // decode autoregressively, so it is refused rather than ignored.
    if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled &&
        info_.draft_depth == 0) {
      loom_text_embed_close(embed_);
      throw std::runtime_error(
          "loom: --speculative " + BackendName(options.speculative) +
          " needs a drafter, and this package has none; give one with " +
          DraftOption(options.speculative));
    }
    loom_text_embed_logits_info(embed_, &logits_);
    chat_config_ = LoadChatConfig(path);
    fingerprint_ = PackageFingerprint(path, preset);
  }
  ~LoomModel() {
    for (auto& [id, checkpoint] : checkpoints_)
      loom_text_embed_checkpoint_release(embed_, checkpoint);
    loom_text_embed_close(embed_);
  }
  LoomModel(const LoomModel&) = delete;
  LoomModel& operator=(const LoomModel&) = delete;

  [[nodiscard]] const loom_text_embed_info_t& info() const { return info_; }
  // The package's chat dialect: the "chat" object of its llm_config.json,
  // when it has one.
  [[nodiscard]] const std::optional<json::Value>& chat_config() const {
    return chat_config_;
  }
  // The logits capability of this load (enabled: rows can be read).
  [[nodiscard]] const loom_text_embed_logits_info_t& logits() const {
    return logits_;
  }
  // The package's manifest and programs (and the preset): persisted state is
  // only valid for the same ones.
  [[nodiscard]] const std::string& Fingerprint() const { return fingerprint_; }
  [[nodiscard]] loom_text_embed_t* embed() const { return embed_; }
  std::recursive_mutex& mutex() const { return mutex_; }

  // Throws on a nonzero status other than `allowed`.
  int Check(int status, const char* what, int allowed = 0) const {
    if (status != 0 && status != allowed)
      throw std::runtime_error(std::string("loom: ") + what + ": " +
                               loom_text_embed_error(embed_));
    return status;
  }

  // Checkpoint handles named by cache payloads.
  std::uint64_t Keep(loom_text_embed_checkpoint_t* checkpoint) {
    const auto id = next_checkpoint_++;
    checkpoints_.emplace(id, checkpoint);
    return id;
  }
  loom_text_embed_checkpoint_t* Find(std::uint64_t id) const {
    const auto found = checkpoints_.find(id);
    return found == checkpoints_.end() ? nullptr : found->second;
  }
  // Releases the oldest retained checkpoint (its cache entry then misses);
  // false when none is left. Used when the device pool cannot back a row.
  bool ReleaseOldest() {
    std::lock_guard lock(mutex_);
    if (checkpoints_.empty())
      return false;
    auto oldest = checkpoints_.begin();
    for (auto it = checkpoints_.begin(); it != checkpoints_.end(); ++it)
      if (it->first < oldest->first)
        oldest = it;
    loom_text_embed_checkpoint_release(embed_, oldest->second);
    checkpoints_.erase(oldest);
    return true;
  }
  void Release(std::uint64_t id) {
    std::lock_guard lock(mutex_);
    const auto found = checkpoints_.find(id);
    if (found == checkpoints_.end())
      return;
    loom_text_embed_checkpoint_release(embed_, found->second);
    checkpoints_.erase(found);
  }

private:
  package::Resolved package_;
  loom_text_embed_t* embed_{nullptr};
  loom_text_embed_info_t info_{};
  loom_text_embed_logits_info_t logits_{};
  std::optional<json::Value> chat_config_;
  mutable std::recursive_mutex mutex_;
  std::unordered_map<std::uint64_t, loom_text_embed_checkpoint_t*> checkpoints_;
  std::uint64_t next_checkpoint_{1};
  std::string fingerprint_;

  // The "chat" object of the package's llm_config.json (the kit's input,
  // which model.json lists so loom-fetch delivers it).
  static std::optional<json::Value> LoadChatConfig(const std::string& path) {
    const auto manifest = ManifestPath(path);
    if (!manifest)
      return std::nullopt;
    std::ifstream in(manifest->parent_path() / "llm_config.json",
                     std::ios::binary);
    if (!in)
      return std::nullopt;
    const std::string text((std::istreambuf_iterator<char>(in)), {});
    try {
      const auto config = json::parse(text);
      const auto* chat = config.find("chat");
      if (chat != nullptr && chat->is_object())
        return *chat;
    } catch (const std::exception&) {
    }
    return std::nullopt;
  }

  // The card declares "llm.logits" (the package carries the readback).
  static void Warn(const std::string& message) {
    ext::Host{}.Log(ext::LogLevel::kWarn, "loom", message);
  }

  static bool DeclaresLogits(const std::string& path) {
    const auto manifest = ManifestPath(path);
    if (!manifest)
      return false;
    std::ifstream in(*manifest, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)), {});
    return text.find("\"llm.logits\"") != std::string::npos;
  }

  static std::string PackageFingerprint(const std::string& path,
                                        const std::string& preset) {
    namespace fs = std::filesystem;
    const fs::path root =
        fs::is_directory(path) ? fs::path(path) : fs::path(path).parent_path();
    std::vector<fs::path> files{root / "model.json"};
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(root / "program", error))
      if (entry.is_regular_file())
        files.push_back(entry.path());
    std::sort(files.begin() + 1, files.end());
    crypto::Sha256Hasher hasher;
    hasher.Update(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(preset.data()), preset.size()));
    for (const auto& file : files) {
      std::ifstream in(file, std::ios::binary);
      const std::string bytes((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
      hasher.Update(std::span<const std::uint8_t>(
          reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
    }
    static constexpr char kHex[] = "0123456789abcdef";
    std::string hex;
    for (const auto byte : hasher.Finish()) {
      hex += kHex[byte >> 4];
      hex += kHex[byte & 15];
    }
    return hex;
  }
};

// One request's output reader (the package's chat program and parser).
class LoomPromptContext final : public ext::TextPromptContext {
public:
  LoomPromptContext(std::shared_ptr<LoomModel> model,
                    loom_text_embed_request_t* request)
      : model(std::move(model)), request(request) {}
  ~LoomPromptContext() override {
    std::lock_guard lock(model->mutex());
    loom_text_embed_request_free(request);
  }
  std::shared_ptr<LoomModel> model;
  loom_text_embed_request_t* request;
  // Input ranges replaced by projected image features; images are projected
  // on the first prefill that needs them.
  std::vector<std::pair<std::size_t, std::size_t>> images;
  mutable bool projected{false};
  // Reasoning tokens allowed before the runner closes reasoning.
  std::optional<std::size_t> reasoning_budget;
  // The prompt ends inside an open reasoning block.
  bool opens_reasoning{false};
  // The request's grammar can constrain this package's raw tokens: its
  // reasoning markers and (when the request declares tools) its native
  // tool-call syntax are the ones Gufo's grammars are written for.
  bool constraint_ok{false};
  // The marker that starts a native tool call (a lazy tool grammar restricts
  // once it is written).
  std::string lazy_open;
  // The request declares tools with tool_choice auto and no response format:
  // its grammar is lazy, free until the call marker has been written.
  bool lazy_tools{false};
  [[nodiscard]] bool has_images() const { return !images.empty(); }
};

}  // namespace gufo::loom

#endif  // GUFO_LOOM_MODEL_HPP_
