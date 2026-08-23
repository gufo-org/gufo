#pragma once

// Shared test utilities for the Qwen module (L1) e2e tests.
//
// Purely additive Phase 1. No dependency on HIP headers so that a CPU-only
// module test binary can include this file. HIP callers do their own stream
// synchronization by passing a `post` sync hook to the timing helpers.
//
// Everything here is header-only `inline` (no CMake wiring yet).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/testing/compare/logit_comparator.hpp"

namespace strix::test {

// ---------------------------------------------------------------------------
// Seeded RNG
// ---------------------------------------------------------------------------

inline std::mt19937 make_seeded_rng(std::uint32_t seed = 0x5EED5u) {
  return std::mt19937(seed);
}

// ---------------------------------------------------------------------------
// Tensor builders (row-major float, hosted in std::vector<float>)
// ---------------------------------------------------------------------------

inline std::vector<float> make_constant_tensor(std::size_t n, float v = 0.0F) {
  return std::vector<float>(n, v);
}

inline std::vector<float> make_random_tensor(std::size_t n, std::mt19937& rng,
                                             float lo = -1.0F, float hi = 1.0F) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> out(n);
  for (float& x : out) x = dist(rng);
  return out;
}

// Shaped tensor (rows x cols, row-major).
inline std::vector<float> make_tensor(std::size_t rows, std::size_t cols,
                                      std::mt19937& rng, float lo = -1.0F,
                                      float hi = 1.0F) {
  return make_random_tensor(rows * cols, rng, lo, hi);
}

inline std::size_t mt_index(std::size_t row, std::size_t col,
                            std::size_t cols) {
  return row * cols + col;
}

inline std::string shape_str(std::size_t rows, std::size_t cols) {
  return std::to_string(rows) + "x" + std::to_string(cols);
}

// ---------------------------------------------------------------------------
// Timing helper
//
// measure_time(fn, reps, warmup) times N repetitions after a warmup run and
// returns mean/median/p99/min/max (milliseconds). A `post` hook (e.g. a
// hipStreamSynchronize trampoline) may be supplied so HIP callers measure a
// fully-synced launch; CPU callers pass the default no-op. Everything is
// wall-clock; it does not require the calling thread to be a HIP-thread.
// ---------------------------------------------------------------------------

struct TimingStats {
  int samples{0};
  double mean_ms{0.0};
  double median_ms{0.0};
  double p99_ms{0.0};
  double min_ms{0.0};
  double max_ms{0.0};
};

inline TimingStats summarize_times(std::vector<double> times) {
  TimingStats s;
  s.samples = static_cast<int>(times.size());
  if (times.empty()) return s;

  std::vector<double> sorted = times;
  std::sort(sorted.begin(), sorted.end());

  double sum = 0.0;
  for (double v : sorted) sum += v;
  s.mean_ms = sum / static_cast<double>(sorted.size());

  auto median = [](const std::vector<double>& v) {
    std::size_t n = v.size();
    if (n == 0) return 0.0;
    if (n % 2 == 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
  };
  s.median_ms = median(sorted);

  std::size_t p99_idx = static_cast<std::size_t>(
      std::ceil(0.99 * static_cast<double>(sorted.size())) - 1.0);
  if (p99_idx >= sorted.size()) p99_idx = sorted.size() - 1;
  s.p99_ms = sorted[p99_idx];

  s.min_ms = sorted.front();
  s.max_ms = sorted.back();
  return s;
}

template <class F>
TimingStats measure_time(F&& fn, int reps = 20, int warmup = 3) {
  return measure_time(fn, reps, warmup, [] {});
}

template <class F, class Sync>
TimingStats measure_time(F&& fn, int reps, int warmup, Sync&& post) {
  using clock = std::chrono::steady_clock;
  for (int i = 0; i < warmup; ++i) {
    fn();
    post();
  }
  std::vector<double> times;
  times.reserve(static_cast<std::size_t>(reps));
  for (int i = 0; i < reps; ++i) {
    auto t0 = clock::now();
    fn();
    post();
    auto t1 = clock::now();
    times.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return summarize_times(std::move(times));
}

// ---------------------------------------------------------------------------
// Baseline record: per-module timing baseline (<artifact>.json)
// ---------------------------------------------------------------------------

struct BaselineRecord {
  double time_us{0.0};
  std::string model;
  std::string config_hash;
};

// FNV-1a 64-bit over a string; used to build a stable config hash.
inline std::uint64_t fnv1a64(std::string_view s) {
  const std::uint64_t prime = 14695981039346656037ULL;
  const std::uint64_t offset = 1099511628211ULL;  // 64-bit FNV offset basis
  std::uint64_t h = offset;
  for (unsigned char c : s) {
    h ^= static_cast<std::uint64_t>(c);
    h *= prime;
  }
  return h;
}

inline std::string compute_config_hash(std::string_view model,
                                       std::string_view signature) {
  std::string joined;
  joined.reserve(model.size() + signature.size());
  joined.append(model);
  joined.push_back('|');
  joined.append(signature);
  return std::to_string(fnv1a64(joined));
}

// Minimal flat JSON object reader: returns an ordered list of (key, value)
// pairs for a top-level { ... } object. Values are kept as raw tokens
// (numbers kept verbatim; strings unescaped minus the surrounding quotes).
// This is intentionally tiny and only shapes itself to the baseline record.
inline std::vector<std::pair<std::string, std::string>> parse_flat_json(
    std::string_view txt) {
  std::vector<std::pair<std::string, std::string>> out;
  std::size_t i = 0;

  auto skip_ws = [&] {
    while (i < txt.size() &&
           (txt[i] == ' ' || txt[i] == '\t' || txt[i] == '\n' ||
            txt[i] == '\r')) {
      ++i;
    }
  };
  auto parse_string = [&](std::string& dst) -> bool {
    // assumes txt[i] == '"'
    ++i;  // consume opening quote
    dst.clear();
    while (i < txt.size() && txt[i] != '"') {
      if (txt[i] == '\\' && i + 1 < txt.size()) {
        ++i;
        dst.push_back(txt[i]);
        ++i;
      } else {
        dst.push_back(txt[i]);
        ++i;
      }
    }
    if (i >= txt.size()) return false;
    ++i;  // consume closing quote
    return true;
  };

  skip_ws();
  if (i >= txt.size() || txt[i] != '{') return out;
  ++i;  // consume '{'

  while (true) {
    skip_ws();
    if (i >= txt.size()) break;
    if (txt[i] == '}') {
      ++i;
      break;
    }
    if (txt[i] == ',') {
      ++i;
      continue;
    }
    if (txt[i] != '"') break;
    std::string key;
    if (!parse_string(key)) break;
    skip_ws();
    if (i >= txt.size() || txt[i] != ':') break;
    ++i;  // consume ':'
    skip_ws();

    std::string val;
    if (i < txt.size() && txt[i] == '"') {
      if (!parse_string(val)) break;
    } else {
      // bare token (number / true / false / null) up to ",", "}" or whitespace
      while (i < txt.size() && txt[i] != ',' && txt[i] != '}' &&
             txt[i] != ' ' && txt[i] != '\t' && txt[i] != '\n' &&
             txt[i] != '\r') {
        val.push_back(txt[i]);
        ++i;
      }
    }
    out.emplace_back(std::move(key), std::move(val));
    skip_ws();
    if (i < txt.size() && txt[i] == ',') {
      ++i;
      continue;
    }
    if (i < txt.size() && txt[i] == '}') {
      ++i;
      break;
    }
  }
  return out;
}

inline std::optional<BaselineRecord> load_baseline(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (!in) return std::nullopt;
  std::ostringstream ss;
  ss << in.rdbuf();
  auto kv = parse_flat_json(ss.str());
  if (kv.empty()) return std::nullopt;

  BaselineRecord rec;
  bool have_time = false;
  for (const auto& [k, v] : kv) {
    if (k == "time_us") {
      rec.time_us = std::stod(v);
      have_time = true;
    } else if (k == "model") {
      rec.model = v;
    } else if (k == "config_hash") {
      rec.config_hash = v;
    }
  }
  if (!have_time) return std::nullopt;
  return rec;
}

inline bool save_baseline(std::string_view path, const BaselineRecord& rec) {
  std::filesystem::path p{std::string(path)};
  if (p.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
    if (ec) return false;
  }
  std::ofstream out(std::string(path), std::ios::trunc);
  if (!out) return false;
  out << "{\n"
      << "  \"time_us\": " << rec.time_us << ",\n"
      << "  \"model\": \"" << rec.model << "\",\n"
      << "  \"config_hash\": \"" << rec.config_hash << "\"\n"
      << "}\n";
  return static_cast<bool>(out);
}

// ---------------------------------------------------------------------------
// CompareLogits wrapper (reuses strix::testing::CompareLogits)
// ---------------------------------------------------------------------------

struct ModuleCompareResult {
  bool match{false};
  bool finite{false};
  bool top1_match{false};
  float max_abs_diff{0.0F};
  float max_rel_diff{0.0F};
  float mean_abs_diff{0.0F};
  float rmse{0.0F};
  float cosine{0.0F};
  std::size_t first_mismatch_idx{0};
  std::string summary;
};

inline ModuleCompareResult compare_module_logits(
    std::span<const float> reference, std::span<const float> candidate,
    float atol = 1e-3F, float rtol = 1e-3F) {
  strix::testing::LogitCompareResult r =
      strix::testing::CompareLogits(reference, candidate, atol, rtol);
  ModuleCompareResult out;
  out.match = r.match;
  out.finite = r.finite;
  out.top1_match = r.top1_match;
  out.max_abs_diff = r.max_abs_diff;
  out.max_rel_diff = r.max_rel_diff;
  out.mean_abs_diff = r.mean_abs_diff;
  out.rmse = r.root_mean_square_error;
  out.cosine = r.cosine_similarity;
  out.first_mismatch_idx = r.first_mismatch_idx;

  std::ostringstream s;
  s << "match=" << (r.match ? "yes" : "no")
    << " finite=" << (r.finite ? "yes" : "no")
    << " top1=" << (r.top1_match ? "yes" : "no")
    << " max_abs=" << r.max_abs_diff << " mean_abs=" << r.mean_abs_diff
    << " rmse=" << r.root_mean_square_error
    << " cosine=" << r.cosine_similarity;
  out.summary = s.str();
  return out;
}

// Renders the §3 L1 per-module result line.
//   MODULE <name>: PASS correctness=<summary> time=<us> us baseline=<us> us
//   delta=<+/-/>%
inline std::string format_module_result(std::string_view name, bool pass,
                                        std::string_view correctness,
                                        double time_us, double baseline_us) {
  double delta_pct = 0.0;
  if (baseline_us > 0.0) {
    delta_pct = (time_us - baseline_us) / baseline_us * 100.0;
  }
  std::ostringstream s;
  s << "MODULE " << name << ": " << (pass ? "PASS" : "FAIL")
    << " correctness=" << correctness << "  time=" << time_us
    << " us  baseline=" << baseline_us << " us  delta=" << delta_pct
    << "%";
  return s.str();
}

}  // namespace strix::test
