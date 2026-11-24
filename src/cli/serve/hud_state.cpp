// Engine-native observability for the `/hud` page. The request ring is fed
// by RecordRequestMetrics (the single completion choke point) and the node
// telemetry is read straight from sysfs, so the HUD needs no external
// scrape source: everything is data the inference process already owns.
#include "src/cli/serve/hud_state.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>

#include "src/cli/serve/generation_metrics.hpp"
#include "src/core/diagnostics/linux_sysfs.h"

namespace gufo::server {
namespace {

constexpr std::size_t kHudHistoryPoints = 900;  // one hour at 4 s polling
constexpr std::int64_t kHudWindowMs = 30LL * 60 * 1000;

std::int64_t ProcessStartMs() {
  static const std::int64_t start = hud_detail::NowMs();
  return start;
}

/// Sampled series point for the HUD sparklines.
struct SamplePoint {
  std::int64_t ts{0};  // unix seconds
  double decode_tps{0.0};
  double cpu_pct{0.0};
  double disk_read_bps{0.0};
};

/// Deltas between successive `/hud/api/state` polls: live decode/prefill
/// rates from the process-wide token counters, host CPU and whole-disk I/O
/// from /proc.
struct Sampler {
  std::int64_t last_ms{0};
  std::uint64_t last_gen_tokens{0};
  std::uint64_t last_prompt_tokens{0};
  std::uint64_t last_cpu_busy{0};
  std::uint64_t last_cpu_total{0};
  std::uint64_t last_disk_read_sectors{0};
  std::uint64_t last_disk_write_sectors{0};
  std::vector<SamplePoint> history;
};

Sampler& HudSampler() {
  static Sampler sampler;
  return sampler;
}

bool ReadProcStatCpu(std::uint64_t& busy, std::uint64_t& total) {
  std::ifstream file("/proc/stat");
  std::string line;
  if (!std::getline(file, line) || line.rfind("cpu ", 0) != 0) {
    return false;
  }
  std::uint64_t fields[8] = {0};
  std::istringstream stream(line.substr(4));
  for (int i = 0; i < 8; ++i) {
    if (!(stream >> fields[i])) {
      return false;
    }
  }
  total = 0;
  for (int i = 0; i < 8; ++i) {
    total += fields[i];
  }
  busy = total - (fields[3] + fields[4]);  // all minus idle and iowait
  return true;
}

/// Whole disks only (nvme0n1, sda): summing partitions would double count.
bool ReadProcDiskSectors(std::uint64_t& read_sectors,
                         std::uint64_t& write_sectors) {
  read_sectors = 0;
  write_sectors = 0;
  auto is_whole_disk = [](const std::string& name) {
    if (name.rfind("nvme", 0) == 0) {
      // nvme0n1 but not nvme0n1p1.
      return name.find('p') == std::string::npos;
    }
    return (name.size() == 3 && (name[0] == 's' || name[0] == 'h') &&
            name[1] == 'd');
  };
  std::ifstream file("/proc/diskstats");
  std::uint64_t major, minor;
  std::string name;
  std::uint64_t values[8];
  while (file >> major >> minor >> name) {
    if (!is_whole_disk(name)) {
      // Skip the rest of the line.
      std::string rest;
      std::getline(file, rest);
      continue;
    }
    for (int i = 0; i < 8; ++i) {
      values[i] = 0;
      file >> values[i];
    }
    if (!file) {
      return false;
    }
    // Fields after the name: reads(0) merged(1) sectors(2) ms(3)
    // writes(4) merged(5) sectors(6) ms(7).
    read_sectors += values[2];
    write_sectors += values[6];
  }
  return true;
}

struct GpuTelemetry {
  double busy_pct{-1.0};
  std::uint64_t gtt_used{0};
  std::uint64_t gtt_total{0};
  double temperature_c{-1.0};
  double power_w{-1.0};
  double power_avg_w{-1.0};
  double link_gts{-1.0};
  std::uint32_t link_width{0};
};

/// The on-package GPU is the drm card that exposes both GTT accounting and
/// busy percent (virtual display cards expose neither).
std::optional<std::filesystem::path> FindApuDeviceDir() {
  std::error_code ec;
  for (const auto& entry :
       std::filesystem::directory_iterator("/sys/class/drm", ec)) {
    if (ec) {
      break;
    }
    const auto name = entry.path().filename().string();
    if (!name.starts_with("card") || name.find('-') != std::string::npos) {
      continue;
    }
    const auto device = entry.path() / "device";
    if (std::filesystem::exists(device / "mem_info_gtt_used", ec) &&
        std::filesystem::exists(device / "gpu_busy_percent", ec)) {
      return device;
    }
  }
  return std::nullopt;
}

std::optional<std::uint64_t> ReadSysfsNumber(
    const gufo::diagnostics::LinuxSysfs& sysfs,
    const std::filesystem::path& path) {
  const auto text = sysfs.ReadFile(path);
  if (!text) {
    return std::nullopt;
  }
  const auto end = text->find_first_not_of("0123456789");
  try {
    return std::stoull(
        text->substr(0, end == std::string::npos ? text->size() : end));
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<double> ReadSysfsFloat(const gufo::diagnostics::LinuxSysfs& sysfs,
                                     const std::filesystem::path& path) {
  const auto text = sysfs.ReadFile(path);
  if (!text) {
    return std::nullopt;
  }
  try {
    return std::stod(text->substr(0, text->find_first_not_of("0123456789.-")));
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

GpuTelemetry ReadGpuTelemetry(const gufo::diagnostics::LinuxSysfs& sysfs) {
  GpuTelemetry gpu;
  const auto device = FindApuDeviceDir();
  if (!device) {
    return gpu;
  }
  if (auto busy = ReadSysfsNumber(sysfs, *device / "gpu_busy_percent")) {
    gpu.busy_pct = static_cast<double>(*busy);
  }
  if (auto gtt = ReadSysfsNumber(sysfs, *device / "mem_info_gtt_used")) {
    gpu.gtt_used = *gtt;
  }
  if (auto gtt = ReadSysfsNumber(sysfs, *device / "mem_info_gtt_total")) {
    gpu.gtt_total = *gtt;
  }
  if (auto link = ReadSysfsFloat(sysfs, *device / "current_link_speed")) {
    gpu.link_gts = *link;
  }
  if (auto width = ReadSysfsNumber(sysfs, *device / "current_link_width")) {
    gpu.link_width = static_cast<std::uint32_t>(*width);
  }
  // hwmon lives one level down: device/hwmon/hwmon*/{temp1_input,...}.
  std::error_code ec;
  const auto hwmon_base = *device / "hwmon";
  if (std::filesystem::exists(hwmon_base, ec)) {
    for (const auto& hwmon_entry :
         std::filesystem::directory_iterator(hwmon_base, ec)) {
      if (ec) {
        break;
      }
      if (auto temp =
              ReadSysfsNumber(sysfs, hwmon_entry.path() / "temp1_input")) {
        gpu.temperature_c = static_cast<double>(*temp) / 1000.0;
      }
      if (auto power =
              ReadSysfsNumber(sysfs, hwmon_entry.path() / "power1_input")) {
        gpu.power_w = static_cast<double>(*power) / 1e6;
      }
      if (auto power =
              ReadSysfsNumber(sysfs, hwmon_entry.path() / "power1_average")) {
        gpu.power_avg_w = static_cast<double>(*power) / 1e6;
      }
      break;
    }
  }
  return gpu;
}

}  // namespace

json::Value HudStateJson(const TextGenerationBackend& backend) {
  std::lock_guard lock(hud_detail::RequestsMutex());
  const std::int64_t now_ms = hud_detail::NowMs();
  static const gufo::diagnostics::LinuxSysfs sysfs;

  // Live rates: token counter deltas over the poll interval, host CPU and
  // disk from /proc. The first poll has no baseline and reports nulls.
  Sampler& sampler = HudSampler();
  double decode_now = 0.0;
  bool decode_active = false;
  double prefill_now = 0.0;
  bool prefill_active = false;
  double cpu_pct = 0.0;
  bool cpu_ok = false;
  double disk_read_bps = 0.0;
  double disk_write_bps = 0.0;
  bool disk_ok = false;
  if (sampler.last_ms > 0 && now_ms > sampler.last_ms) {
    const double dt = (now_ms - sampler.last_ms) / 1000.0;
    if (dt >= 0.05) {
      const auto gen = detail::TotalGenTokens().load(std::memory_order_relaxed);
      const auto prompt =
          detail::TotalPromptTokens().load(std::memory_order_relaxed);
      if (gen > sampler.last_gen_tokens) {
        decode_now = static_cast<double>(gen - sampler.last_gen_tokens) / dt;
        decode_active = true;
      }
      if (prompt > sampler.last_prompt_tokens) {
        prefill_now =
            static_cast<double>(prompt - sampler.last_prompt_tokens) / dt;
        prefill_active = true;
      }
      std::uint64_t busy = 0;
      std::uint64_t total = 0;
      if (ReadProcStatCpu(busy, total) && total > sampler.last_cpu_total) {
        cpu_pct = 100.0 * static_cast<double>(busy - sampler.last_cpu_busy) /
                  static_cast<double>(total - sampler.last_cpu_total);
        cpu_ok = true;
      }
      std::uint64_t read_sectors = 0;
      std::uint64_t write_sectors = 0;
      if (ReadProcDiskSectors(read_sectors, write_sectors)) {
        disk_read_bps =
            read_sectors > sampler.last_disk_read_sectors
                ? static_cast<double>(read_sectors -
                                      sampler.last_disk_read_sectors) *
                      512.0 / dt
                : 0.0;
        disk_write_bps =
            write_sectors > sampler.last_disk_write_sectors
                ? static_cast<double>(write_sectors -
                                      sampler.last_disk_write_sectors) *
                      512.0 / dt
                : 0.0;
        disk_ok = true;
      }
    }
  }
  sampler.last_ms = now_ms;
  sampler.last_gen_tokens =
      detail::TotalGenTokens().load(std::memory_order_relaxed);
  sampler.last_prompt_tokens =
      detail::TotalPromptTokens().load(std::memory_order_relaxed);
  {
    std::uint64_t busy = 0;
    std::uint64_t total = 0;
    if (ReadProcStatCpu(busy, total)) {
      sampler.last_cpu_busy = busy;
      sampler.last_cpu_total = total;
    }
    std::uint64_t read_sectors = 0;
    std::uint64_t write_sectors = 0;
    if (ReadProcDiskSectors(read_sectors, write_sectors)) {
      sampler.last_disk_read_sectors = read_sectors;
      sampler.last_disk_write_sectors = write_sectors;
    }
  }
  if (decode_active) {
    sampler.history.push_back({now_ms / 1000, decode_now,
                               cpu_ok ? cpu_pct : 0.0,
                               disk_ok ? disk_read_bps : 0.0});
  }
  if (sampler.history.size() > kHudHistoryPoints) {
    sampler.history.resize(kHudHistoryPoints);
  }

  // Model and session context.
  const auto sessions = backend.session_states();
  const std::uint32_t context = backend.max_context();
  std::size_t used_tokens = 0;
  bool speculative = false;
  std::size_t reading_sessions = 0;
  std::size_t generating_sessions = 0;
  for (const auto& session : sessions) {
    used_tokens += session.prompt_tokens + session.generated_tokens;
    speculative = speculative || session.speculative;
    if (session.processing) {
      // Prompt still being prefilled (cached tokens never go through the
      // processor), or generating once it is fully read.
      if (session.processed_prompt_tokens + session.cached_prompt_tokens <
          session.prompt_tokens) {
        ++reading_sessions;
      } else {
        ++generating_sessions;
      }
    }
  }
  double kv_pct = 0.0;
  const std::uint64_t capacity =
      static_cast<std::uint64_t>(sessions.size()) * context;
  if (capacity > 0) {
    kv_pct = 100.0 * static_cast<double>(used_tokens) /
             static_cast<double>(capacity);
  }

  // Recent request ring, newest first.
  json::Value requests = json::Value::array();
  for (const auto& record : hud_detail::RequestsRing()) {
    json::Value row = json::Value::object();
    row["ts"] = record.ts_ms / 1000;
    row["status"] = record.status;
    row["prompt"] = static_cast<std::int64_t>(record.prompt_tokens);
    row["cached"] = static_cast<std::int64_t>(record.cached_tokens);
    row["out"] = static_cast<std::int64_t>(record.completion_tokens);
    row["prefill_tps"] = record.prefill_tps;
    row["decode_tps"] = record.decode_tps;
    row["ttft_ms"] = record.ttft_ms;
    row["dur"] = record.duration_ms;
    requests.push_back(std::move(row));
  }

  // 30-minute window over the ring for averages and min/max.
  const std::int64_t window_start = now_ms - kHudWindowMs;
  double decode_sum = 0.0;
  double prefill_sum = 0.0;
  double decode_min = -1.0;
  double decode_max = 0.0;
  double prefill_min = -1.0;
  double prefill_max = 0.0;
  std::uint64_t window_prompt = 0;
  std::uint64_t window_cached = 0;
  std::size_t window_count = 0;
  for (const auto& record : hud_detail::RequestsRing()) {
    if (record.ts_ms < window_start) {
      continue;
    }
    ++window_count;
    window_prompt += record.prompt_tokens;
    window_cached += record.cached_tokens;
    if (record.decode_tps > 0.0) {
      decode_sum += record.decode_tps;
      decode_min = decode_min < 0 ? record.decode_tps
                                  : std::min(decode_min, record.decode_tps);
      decode_max = std::max(decode_max, record.decode_tps);
    }
    if (record.prefill_tps > 0.0) {
      prefill_sum += record.prefill_tps;
      prefill_min = prefill_min < 0 ? record.prefill_tps
                                    : std::min(prefill_min, record.prefill_tps);
      prefill_max = std::max(prefill_max, record.prefill_tps);
    }
  }
  const auto draft_tokens =
      detail::TotalDraftTokens().load(std::memory_order_relaxed);
  const auto draft_accepted =
      detail::TotalDraftAcceptedTokens().load(std::memory_order_relaxed);
  const auto lost = detail::DeviceLostTotal().load(std::memory_order_relaxed);

  // Node telemetry, engine-native.
  const auto gpu = ReadGpuTelemetry(sysfs);
  const auto mem = sysfs.QueryMemInfo();

  json::Value prom = json::Value::object();
  prom["decode_now"] = decode_active ? json::Value(decode_now) : json::Value();
  prom["prefill_now"] =
      prefill_active ? json::Value(prefill_now) : json::Value();
  prom["decode_avg"] =
      window_count > 0 ? json::Value(decode_sum / window_count) : json::Value();
  prom["decode_min"] = decode_min > 0 ? json::Value(decode_min) : json::Value();
  prom["decode_max"] = decode_max > 0 ? json::Value(decode_max) : json::Value();
  prom["prefill_avg"] = window_count > 0
                            ? json::Value(prefill_sum / window_count)
                            : json::Value();
  prom["prefill_min"] =
      prefill_min > 0 ? json::Value(prefill_min) : json::Value();
  prom["prefill_max"] =
      prefill_max > 0 ? json::Value(prefill_max) : json::Value();
  prom["prompt_reuse"] =
      window_prompt > 0
          ? json::Value(100.0 * static_cast<double>(window_cached) /
                        static_cast<double>(window_prompt))
          : json::Value();
  prom["spec_accept"] =
      draft_tokens > 0
          ? json::Value(100.0 * static_cast<double>(draft_accepted) /
                        static_cast<double>(draft_tokens))
          : json::Value();
  prom["device_lost"] = static_cast<std::int64_t>(lost);
  prom["processing"] =
      detail::RequestsProcessing().load(std::memory_order_relaxed);
  prom["deferred"] = detail::RequestsDeferred().load(std::memory_order_relaxed);
  prom["reading"] = static_cast<std::int64_t>(reading_sessions);
  prom["generating"] = static_cast<std::int64_t>(generating_sessions);
  prom["gpu_busy"] =
      gpu.busy_pct >= 0 ? json::Value(gpu.busy_pct) : json::Value();
  prom["gtt_used"] = static_cast<std::int64_t>(gpu.gtt_used);
  prom["gtt_total"] = static_cast<std::int64_t>(gpu.gtt_total);
  prom["gpu_temp"] =
      gpu.temperature_c >= 0 ? json::Value(gpu.temperature_c) : json::Value();
  prom["power_w"] = gpu.power_w >= 0 ? json::Value(gpu.power_w) : json::Value();
  prom["power_avg_w"] =
      gpu.power_avg_w >= 0 ? json::Value(gpu.power_avg_w) : json::Value();
  prom["pcie_gts"] =
      gpu.link_gts >= 0 ? json::Value(gpu.link_gts) : json::Value();
  prom["pcie_width"] = static_cast<std::int64_t>(gpu.link_width);
  prom["cpu_pct"] = cpu_ok ? json::Value(cpu_pct) : json::Value();
  prom["disk_read"] = disk_ok ? json::Value(disk_read_bps) : json::Value();
  prom["disk_write"] = disk_ok ? json::Value(disk_write_bps) : json::Value();
  if (mem) {
    prom["ram_total"] = static_cast<std::int64_t>(mem->total_bytes);
    prom["ram_used"] =
        mem->total_bytes > mem->available_bytes
            ? static_cast<std::int64_t>(mem->total_bytes - mem->available_bytes)
            : std::int64_t{0};
  } else {
    prom["ram_total"] = json::Value();
    prom["ram_used"] = json::Value();
  }
  prom["kv_pct"] = kv_pct;
  prom["n_tokens_max"] = static_cast<std::int64_t>(
      detail::MaxSequenceTokens().load(std::memory_order_relaxed));

  json::Value model = json::Value::object();
  model["served"] = backend.model_id();
  model["context"] = static_cast<std::int64_t>(context);
  model["sessions"] = static_cast<std::int64_t>(sessions.size());
  model["spec"] = speculative ? "MTP" : json::Value();
  model["started"] = ProcessStartMs();

  json::Value ranges = json::Value::object();
  for (const auto& key : {std::pair<const std::string, int>{"decode", 1},
                          {"cpu_pct", 2},
                          {"disk_read", 3}}) {
    json::Value series = json::Value::array();
    for (const auto& point : sampler.history) {
      const double value = key.second == 1   ? point.decode_tps
                           : key.second == 2 ? point.cpu_pct
                                             : point.disk_read_bps;
      json::Value pair = json::Value::array();
      pair.push_back(point.ts);
      pair.push_back(value);
      series.push_back(std::move(pair));
    }
    ranges[key.first] = std::move(series);
  }

  json::Value root = json::Value::object();
  root["prom"] = std::move(prom);
  root["model"] = std::move(model);
  root["requests"] = std::move(requests);
  root["ranges"] = std::move(ranges);
  return root;
}

}  // namespace gufo::server
