#include "src/cli/model/download.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/fs.hpp"
#include "src/core/platform/os_info.hpp"

namespace gufo::cli::model_download {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kFlushBytes = 8U << 20U;
constexpr int kMaximumAttempts = 10;

std::string Env(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr ? value : "";
}

fs::path HfHome() {
  if (const auto home = Env("HF_HOME"); !home.empty())
    return home;
  const auto cache = platform::UserCacheDirectory();
  return cache.empty() ? fs::path{} : cache / "huggingface";
}

std::string ReadFirstLine(const fs::path& path) {
  std::ifstream input(path);
  std::string line;
  std::getline(input, line);
  while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
    line.pop_back();
  return line;
}

std::string Escape(CURL* curl, std::string_view path) {
  std::string escaped;
  std::size_t start = 0;
  while (start <= path.size()) {
    const auto end = std::min(path.find('/', start), path.size());
    const auto segment = path.substr(start, end - start);
    char* part = curl_easy_escape(curl, segment.data(),
                                  static_cast<int>(segment.size()));
    if (part == nullptr)
      throw std::runtime_error("cannot escape " + std::string(path));
    escaped += part;
    curl_free(part);
    if (end < path.size())
      escaped += '/';
    start = end + 1;
  }
  return escaped;
}

std::string Bytes(std::uint64_t bytes) {
  std::array<char, 32> text{};
  const double value = static_cast<double>(bytes);
  if (value >= 1e9)
    std::snprintf(text.data(), text.size(), "%.2f GB", value / 1e9);
  else
    std::snprintf(text.data(), text.size(), "%.1f MB", value / 1e6);
  return text.data();
}

struct Chunk {
  std::uint64_t begin = 0;
  std::uint64_t end = 0;     // exclusive
  std::uint64_t cursor = 0;  // next byte to write
  bool done = false;
  bool active = false;
  int attempts = 0;
  Clock::time_point not_before{};
};

struct Job {
  Item item;
  fs::path part;
  fs::path progress;
  std::string resolve_url;
  std::string url;  // CDN location of a ranged file, once resolved
  bool ranged = false;
  bool resolving = false;
  int resolve_attempts = 0;
  Clock::time_point resolve_not_before{};
  std::vector<Chunk> chunks;
  std::size_t done = 0;
  int active = 0;
  std::unique_ptr<std::fstream> out;
  std::future<std::string> hash;
  bool hashing = false;
  bool finished = false;
  bool verified_once = false;
};

class Downloader;

struct Transfer {
  Downloader* owner = nullptr;
  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> easy{nullptr,
                                                           curl_easy_cleanup};
  Job* job = nullptr;
  Chunk* chunk = nullptr;  // null while resolving
  std::vector<char> buffer;
  bool rejected = false;    // wrong status or more bytes than requested
  std::uint64_t total = 0;  // Content-Range total, while resolving
  std::array<char, CURL_ERROR_SIZE> error{};
};

class Downloader {
public:
  Downloader(const fs::path& store, const Options& options)
      : store_(store),
        options_(options),
        endpoint_(Endpoint()),
        token_(Token()),
        multi_(curl_multi_init(), curl_multi_cleanup),
        terminal_(platform::StderrIsTerminal()) {
    if (!multi_)
      throw std::runtime_error("cannot initialize libcurl");
    // One connection per transfer: multiplexed HTTP/2 streams would share
    // one TCP window.
    curl_multi_setopt(multi_.get(), CURLMOPT_PIPELINING, CURLPIPE_NOTHING);
    curl_multi_setopt(multi_.get(), CURLMOPT_MAX_TOTAL_CONNECTIONS,
                      static_cast<long>(options_.connections));
    protocols_ = endpoint_.starts_with("http://") ? "http,https" : "https";
  }

  void Run(const std::vector<Item>& items) {
    fs::create_directories(store_ / "blobs");
    fs::create_directories(store_ / "downloads");
    std::uint64_t needed = 0;
    for (const auto& item : items) {
      if (Present(item))
        continue;
      auto& job = jobs_.emplace_back();
      job.item = item;
      job.part = store_ / "downloads" / (item.sha256 + ".part");
      job.progress = store_ / "downloads" / (item.sha256 + ".progress");
      job.resolve_url = endpoint_ + "/" + item.repo + "/resolve/" +
                        item.revision + "/" + Escape(Scratch(), item.path);
      if (Reuse(job))
        continue;
      Prepare(job);
      for (const auto& chunk : job.chunks)
        needed += chunk.end - chunk.cursor;
    }
    if (jobs_.empty())
      return;
    if (reused_files_ > 0)
      std::cerr << "Reusing " << reused_files_ << " files ("
                << Bytes(reused_bytes_) << ") from " << HubCache().string()
                << "\n";
    const auto space = fs::space(store_);
    if (space.available < needed)
      throw std::runtime_error("downloading needs " + Bytes(needed) + ", but " +
                               store_.string() + " has " +
                               Bytes(space.available) + " free");
    total_ = needed;
    started_ = Clock::now();
    reported_ = started_;
    if (needed > 0)
      std::cerr << "Downloading " << Bytes(needed) << " over "
                << options_.connections << " connections\n";
    Loop();
    Report(true);
  }

private:
  // Already in the store: blobs are only ever written after verification.
  bool Present(const Item& item) {
    std::error_code ec;
    const auto blob = store_ / "blobs" / item.sha256;
    return fs::is_regular_file(blob, ec) &&
           fs::file_size(blob, ec) == item.size;
  }

  // Links a matching blob of the Hugging Face cache instead of downloading
  // it; it is verified like a download.
  bool Reuse(Job& job) {
    const auto hub = HubCache();
    if (hub.empty())
      return false;
    const auto& repo = job.item.repo;
    const auto slash = repo.find('/');
    const auto cached =
        hub /
        ("models--" + repo.substr(0, slash) + "--" + repo.substr(slash + 1)) /
        "blobs" / job.item.sha256;
    std::error_code ec;
    if (!fs::is_regular_file(cached, ec) ||
        fs::file_size(cached, ec) != job.item.size)
      return false;
    fs::remove(job.part, ec);
    fs::remove(job.progress, ec);
    fs::create_hard_link(cached, job.part, ec);
    if (ec) {
      ec.clear();
      fs::copy_file(cached, job.part, ec);
      if (ec)
        return false;
    }
    ++reused_files_;
    reused_bytes_ += job.item.size;
    StartHash(job);
    return true;
  }

  // Splits the file into ranges and recovers the ones a previous run wrote.
  void Prepare(Job& job) {
    const auto size = job.item.size;
    const auto chunk = options_.chunk_bytes;
    job.ranged = size > chunk;
    for (std::uint64_t begin = 0; begin < size; begin += chunk)
      job.chunks.push_back({begin, std::min(size, begin + chunk), begin});
    std::error_code ec;
    bool resumed = false;
    if (fs::is_regular_file(job.part, ec) &&
        fs::file_size(job.part, ec) == size && fs::exists(job.progress, ec)) {
      try {
        std::ifstream input(job.progress, std::ios::binary);
        std::ostringstream text;
        text << input.rdbuf();
        const auto state = json::parse(text.str());
        if (state.member_double("size") == static_cast<double>(size) &&
            state.member_double("chunk") == static_cast<double>(chunk)) {
          if (const auto* done = state.find("done"); done && done->is_array()) {
            for (const auto& index : done->items()) {
              const auto i = index.as_size(job.chunks.size());
              if (i < job.chunks.size() && !job.chunks[i].done) {
                job.chunks[i].done = true;
                job.chunks[i].cursor = job.chunks[i].end;
                ++job.done;
              }
            }
          }
          resumed = true;
        }
      } catch (const std::exception&) {
        resumed = false;
      }
    }
    if (!resumed) {
      for (auto& item : job.chunks) {
        item.done = false;
        item.cursor = item.begin;
      }
      job.done = 0;
      fs::remove(job.progress, ec);
      // A new file: the old one may be a hard link into the Hugging Face
      // cache, which must not be truncated.
      fs::remove(job.part, ec);
      {
        std::ofstream create(job.part, std::ios::binary | std::ios::trunc);
      }
      fs::resize_file(job.part, size);
    } else if (job.done > 0) {
      std::cerr << "Resuming " << job.item.path << "\n";
    }
    if (job.chunks.empty() || job.done == job.chunks.size())
      StartHash(job);
  }

  void SaveProgress(const Job& job) {
    auto done = json::Value::array();
    for (std::size_t i = 0; i < job.chunks.size(); ++i) {
      if (job.chunks[i].done)
        done.push_back(i);
    }
    auto state = json::Value::object();
    state["size"] = job.item.size;
    state["chunk"] = options_.chunk_bytes;
    state["done"] = std::move(done);
    auto temporary = job.progress;
    temporary += ".tmp";
    {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      output << state.dump();
    }
    fs::rename(temporary, job.progress);
  }

  void StartHash(Job& job) {
    job.out.reset();
    job.hashing = true;
    job.hash = std::async(std::launch::async, [path = job.part] {
      return crypto::Sha256FileHex(path);
    });
  }

  void FinishHash(Job& job) {
    job.hashing = false;
    const auto digest = job.hash.get();
    std::error_code ec;
    if (digest == job.item.sha256) {
      fs::rename(job.part, store_ / "blobs" / job.item.sha256);
      fs::remove(job.progress, ec);
      job.finished = true;
      return;
    }
    fs::remove(job.progress, ec);
    if (job.verified_once)
      throw std::runtime_error(job.item.repo + "/" + job.item.path +
                               ": SHA-256 " + digest + " does not match " +
                               job.item.sha256);
    // One fresh download: a resumed or cached copy may have been damaged.
    job.verified_once = true;
    fs::remove(job.part, ec);
    total_ += job.item.size;
    Prepare(job);
  }

  CURL* Scratch() {
    if (!scratch_)
      scratch_.reset(curl_easy_init());
    return scratch_.get();
  }

  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> Easy(
      Transfer& transfer, const std::string& url, bool authenticate) {
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> easy(curl_easy_init(),
                                                             curl_easy_cleanup);
    if (!easy)
      throw std::runtime_error("cannot initialize libcurl");
    auto* curl = easy.get();
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_PRIVATE, &transfer);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, protocols_.c_str());
    curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS_STR, protocols_.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "gufo-model");
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 512L * 1024L);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, transfer.error.data());
    // A bearer token goes only to the endpoint: libcurl drops it when a
    // redirect leaves the host.
    if (authenticate && !token_.empty()) {
      curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BEARER);
      curl_easy_setopt(curl, CURLOPT_XOAUTH2_BEARER, token_.c_str());
    }
    return easy;
  }

  static long Status(CURL* curl) {
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    return status;
  }

  static std::size_t OnHeader(char* data, std::size_t size, std::size_t count,
                              void* user) {
    auto* transfer = static_cast<Transfer*>(user);
    const std::string_view line(data, size * count);
    constexpr std::string_view kRange = "content-range:";
    if (line.size() > kRange.size()) {
      std::string lower(line.substr(0, kRange.size()));
      std::ranges::transform(lower, lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      if (lower == kRange) {
        const auto slash = line.rfind('/');
        if (slash != std::string_view::npos)
          transfer->total = std::strtoull(
              std::string(line.substr(slash + 1)).c_str(), nullptr, 10);
      }
    }
    return size * count;
  }

  // While resolving, one byte proves access and the size; the rest is dropped.
  static std::size_t OnResolveBody(char*, std::size_t size, std::size_t count,
                                   void*) {
    return size * count;
  }

  void Flush(Transfer& transfer) {
    if (transfer.buffer.empty())
      return;
    auto& out = *transfer.job->out;
    out.seekp(static_cast<std::streamoff>(transfer.chunk->cursor));
    out.write(transfer.buffer.data(),
              static_cast<std::streamsize>(transfer.buffer.size()));
    if (!out)
      throw std::runtime_error("cannot write " + transfer.job->part.string());
    transfer.chunk->cursor += transfer.buffer.size();
    done_bytes_ += transfer.buffer.size();
    transfer.buffer.clear();
  }

  static std::size_t OnBody(char* data, std::size_t size, std::size_t count,
                            void* user) {
    auto* transfer = static_cast<Transfer*>(user);
    const auto bytes = size * count;
    auto* chunk = transfer->chunk;
    if (transfer->buffer.empty() && !transfer->rejected) {
      // A server that ignores the range must not overwrite other ranges.
      const long status = Status(transfer->easy.get());
      const bool whole =
          chunk->cursor == 0 && chunk->end == transfer->job->item.size;
      if (status != 206 && !(status == 200 && whole))
        transfer->rejected = true;
    }
    if (transfer->rejected ||
        chunk->cursor + transfer->buffer.size() + bytes > chunk->end) {
      transfer->rejected = true;
      return 0;
    }
    transfer->buffer.insert(transfer->buffer.end(), data, data + bytes);
    if (transfer->buffer.size() >= kFlushBytes) {
      try {
        transfer->owner->Flush(*transfer);
      } catch (const std::exception&) {
        return 0;
      }
    }
    return bytes;
  }

  void StartResolve(Job& job) {
    auto transfer = std::make_unique<Transfer>();
    transfer->job = &job;
    transfer->easy = Easy(*transfer, job.resolve_url, true);
    auto* curl = transfer->easy.get();
    curl_easy_setopt(curl, CURLOPT_RANGE, "0-0");
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, OnHeader);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, transfer.get());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, OnResolveBody);
    job.resolving = true;
    Add(std::move(transfer));
  }

  void StartChunk(Job& job, Chunk& chunk) {
    if (!job.out) {
      job.out = std::make_unique<std::fstream>(
          job.part, std::ios::binary | std::ios::in | std::ios::out);
      if (!*job.out)
        throw std::runtime_error("cannot open " + job.part.string());
    }
    auto transfer = std::make_unique<Transfer>();
    transfer->job = &job;
    transfer->chunk = &chunk;
    // A ranged file goes straight to its CDN location, without the token.
    transfer->easy = job.ranged ? Easy(*transfer, job.url, false)
                                : Easy(*transfer, job.resolve_url, true);
    auto* curl = transfer->easy.get();
    const auto range =
        std::to_string(chunk.cursor) + "-" + std::to_string(chunk.end - 1);
    curl_easy_setopt(curl, CURLOPT_RANGE, range.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, OnBody);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, transfer.get());
    transfer->buffer.reserve(kFlushBytes + (512U << 10U));
    chunk.active = true;
    ++job.active;
    Add(std::move(transfer));
  }

  void Add(std::unique_ptr<Transfer> transfer) {
    transfer->owner = this;
    if (curl_multi_add_handle(multi_.get(), transfer->easy.get()) != CURLM_OK)
      throw std::runtime_error("cannot start a transfer");
    active_.push_back(std::move(transfer));
  }

  // Starts work until every connection is busy, in file order so files
  // complete, and get hashed, one after another.
  void Fill() {
    const auto now = Clock::now();
    for (auto& job : jobs_) {
      if (active_.size() >= options_.connections)
        return;
      if (job.finished || job.hashing || job.done == job.chunks.size())
        continue;
      if (job.ranged && job.url.empty()) {
        if (!job.resolving && now >= job.resolve_not_before)
          StartResolve(job);
        continue;
      }
      for (auto& chunk : job.chunks) {
        if (active_.size() >= options_.connections)
          return;
        if (!chunk.done && !chunk.active && now >= chunk.not_before)
          StartChunk(job, chunk);
      }
    }
  }

  static Clock::time_point Backoff(int attempts) {
    return Clock::now() + std::chrono::seconds(std::min(30, 1 << attempts));
  }

  std::string Describe(const Transfer& transfer, CURLcode code,
                       long status) const {
    const auto& item = transfer.job->item;
    std::string where =
        item.repo + "@" + item.revision.substr(0, 12) + " " + item.path + ": ";
    if (status == 401 || status == 403)
      return where + "access denied (HTTP " + std::to_string(status) +
             "). Accept the model's terms on Hugging Face and set HF_TOKEN.";
    if (status == 404)
      return where + "not found at the pinned revision (HTTP 404)";
    if (status >= 400)
      return where + "HTTP " + std::to_string(status);
    if (code != CURLE_OK)
      return where + (transfer.error[0] != '\0' ? transfer.error.data()
                                                : curl_easy_strerror(code));
    return where + "unexpected response";
  }

  void Complete(Transfer& transfer, CURLcode code) {
    auto& job = *transfer.job;
    const long status = Status(transfer.easy.get());
    if (transfer.chunk == nullptr) {
      job.resolving = false;
      if (code == CURLE_OK && (status == 206 || status == 200)) {
        if (status == 206 && transfer.total != job.item.size)
          throw std::runtime_error(Describe(transfer, code, status) + " is " +
                                   std::to_string(transfer.total) +
                                   " bytes; the lock says " +
                                   std::to_string(job.item.size));
        char* effective = nullptr;
        curl_easy_getinfo(transfer.easy.get(), CURLINFO_EFFECTIVE_URL,
                          &effective);
        job.url = effective != nullptr ? effective : job.resolve_url;
        return;
      }
      if (status == 401 || status == 403 || status == 404 ||
          ++job.resolve_attempts > kMaximumAttempts)
        throw std::runtime_error(Describe(transfer, code, status));
      job.resolve_not_before = Backoff(job.resolve_attempts);
      return;
    }
    auto& chunk = *transfer.chunk;
    chunk.active = false;
    --job.active;
    if (!transfer.rejected)
      Flush(transfer);
    if (code == CURLE_OK && !transfer.rejected && chunk.cursor == chunk.end) {
      chunk.done = true;
      ++job.done;
      job.out->flush();
      SaveProgress(job);
      if (job.done == job.chunks.size() && job.active == 0)
        StartHash(job);
      return;
    }
    if (code == CURLE_OK && !transfer.rejected && status < 400) {
      // A short body without an error: retry the remainder.
    } else if (job.ranged &&
               (status == 401 || status == 403 || status == 410)) {
      // The signed CDN location expired: resolve the file again.
      if (job.url.empty() || ++job.resolve_attempts > kMaximumAttempts)
        throw std::runtime_error(Describe(transfer, code, status));
      job.url.clear();
    } else if (status == 401 || status == 403 || status == 404 ||
               status == 416) {
      throw std::runtime_error(Describe(transfer, code, status));
    }
    if (++chunk.attempts > kMaximumAttempts)
      throw std::runtime_error(Describe(transfer, code, status));
    chunk.not_before = Backoff(chunk.attempts);
  }

  void Report(bool last) {
    if (total_ == 0)
      return;
    const auto now = Clock::now();
    if (!last && now - reported_ < (terminal_ ? std::chrono::milliseconds(250)
                                              : std::chrono::seconds(15)))
      return;
    reported_ = now;
    const double seconds =
        std::chrono::duration<double>(now - started_).count();
    const double rate =
        seconds > 0 ? static_cast<double>(done_bytes_) / seconds : 0.0;
    std::array<char, 64> speed{};
    std::snprintf(speed.data(), speed.size(), "%.0f MB/s", rate / 1e6);
    std::string text;
    if (last) {
      std::array<char, 32> elapsed{};
      std::snprintf(elapsed.data(), elapsed.size(), "%.1f s", seconds);
      text = "Downloaded " + Bytes(done_bytes_) + " in " + elapsed.data() +
             " (" + speed.data() + ")";
    } else {
      text = Bytes(done_bytes_) + " / " + Bytes(total_) + "  " + speed.data();
      if (rate > 0) {
        const auto left = static_cast<long long>(
            static_cast<double>(total_ - std::min(total_, done_bytes_)) / rate);
        text += "  " + std::to_string(left / 60) + "m" +
                std::to_string(left % 60) + "s left";
      }
    }
    if (terminal_)
      std::cerr << "\r\033[K" << text << (last ? "\n" : "") << std::flush;
    else
      std::cerr << text << "\n";
  }

  void Loop() {
    while (true) {
      for (auto& job : jobs_) {
        if (job.hashing && job.hash.wait_for(std::chrono::seconds(0)) ==
                               std::future_status::ready)
          FinishHash(job);
      }
      if (std::ranges::all_of(jobs_,
                              [](const Job& job) { return job.finished; }))
        return;
      Fill();
      int running = 0;
      if (curl_multi_perform(multi_.get(), &running) != CURLM_OK)
        throw std::runtime_error("libcurl transfer loop failed");
      int queued = 0;
      while (CURLMsg* message = curl_multi_info_read(multi_.get(), &queued)) {
        if (message->msg != CURLMSG_DONE)
          continue;
        auto* easy = message->easy_handle;
        const auto code = message->data.result;
        curl_multi_remove_handle(multi_.get(), easy);
        const auto found = std::ranges::find_if(
            active_, [&](const auto& t) { return t->easy.get() == easy; });
        auto transfer = std::move(*found);
        active_.erase(found);
        Complete(*transfer, code);
      }
      Report(false);
      curl_multi_poll(multi_.get(), nullptr, 0, 100, nullptr);
    }
  }

public:
  ~Downloader() {
    for (auto& transfer : active_)
      curl_multi_remove_handle(multi_.get(), transfer->easy.get());
    for (auto& job : jobs_) {
      if (job.hashing)
        job.hash.wait();
    }
  }

private:
  fs::path store_;
  Options options_;
  std::string endpoint_;
  std::string token_;
  std::string protocols_;
  std::unique_ptr<CURLM, decltype(&curl_multi_cleanup)> multi_;
  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> scratch_{
      nullptr, curl_easy_cleanup};
  std::list<Job> jobs_;  // stable addresses for transfers
  std::vector<std::unique_ptr<Transfer>> active_;
  bool terminal_ = false;
  std::size_t reused_files_ = 0;
  std::uint64_t reused_bytes_ = 0;
  std::uint64_t total_ = 0;
  std::uint64_t done_bytes_ = 0;
  Clock::time_point started_{};
  Clock::time_point reported_{};
};
}  // namespace

std::string Endpoint() {
  auto endpoint = Env("HF_ENDPOINT");
  if (endpoint.empty())
    endpoint = "https://huggingface.co";
  while (endpoint.ends_with('/'))
    endpoint.pop_back();
  return endpoint;
}

std::string Token() {
  if (auto token = Env("HF_TOKEN"); !token.empty())
    return token;
  const auto home = HfHome();
  return home.empty() ? std::string{} : ReadFirstLine(home / "token");
}

fs::path HubCache() {
  if (const auto cache = Env("HF_HUB_CACHE"); !cache.empty())
    return cache;
  const auto home = HfHome();
  return home.empty() ? fs::path{} : home / "hub";
}

void Fetch(const std::vector<Item>& items, const fs::path& store,
           const Options& options) {
  Downloader downloader(store, options);
  downloader.Run(items);
}

}  // namespace gufo::cli::model_download
