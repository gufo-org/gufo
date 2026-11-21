#ifndef GUFO_SERVER_MUSIC_API_HPP_
#define GUFO_SERVER_MUSIC_API_HPP_

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/media_jobs.hpp"
#include "src/core/json.hpp"
#include "src/ext/music.hpp"

namespace gufo::server {

struct MusicServiceOptions {
  std::filesystem::path storage_root;
  std::chrono::seconds ttl{std::chrono::hours(1)};
  std::size_t queue_capacity{4};
  /// Public model ID; empty uses the runner's.
  std::string model_id;
  std::vector<std::string> model_aliases;
};

/// `gufo serve music`: music jobs on a `MediaJobStore`, one runner.
class MusicService {
public:
  MusicService(std::unique_ptr<ext::MusicRunner> runner,
               MusicServiceOptions options);

  [[nodiscard]] bool ready() const;
  [[nodiscard]] std::string initialization_error() const;
  [[nodiscard]] const std::string& model_id() const { return model_id_; }
  [[nodiscard]] const std::vector<std::string>& model_aliases() const {
    return aliases_;
  }
  [[nodiscard]] bool serves_model(std::string_view name) const;
  [[nodiscard]] const ext::MusicCapabilities& capabilities() const {
    return capabilities_;
  }
  [[nodiscard]] ext::MusicRunner& runner() { return *runner_; }
  [[nodiscard]] MediaJobStore& store() { return *store_; }

private:
  std::shared_ptr<ext::MusicRunner> runner_;
  ext::MusicCapabilities capabilities_;
  std::string model_id_;
  std::vector<std::string> aliases_;
  std::unique_ptr<MediaJobStore> store_;
};

/// `MusicCapabilities` as the `gufo.music` member of a `/v1/models` entry.
[[nodiscard]] json::Value MusicCapabilitiesJson(
    const ext::MusicCapabilities& capabilities);

[[nodiscard]] bool IsMusicApiPath(std::string_view path) noexcept;
[[nodiscard]] HttpResponse HandleMusicApiRequest(const HttpRequest& request,
                                                 MusicService& service);

}  // namespace gufo::server

#endif  // GUFO_SERVER_MUSIC_API_HPP_
