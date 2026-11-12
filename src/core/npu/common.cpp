// The NPU code that needs no XRT: the sysfs probe, instruction patching and
// IRON dispatch libraries.

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <system_error>
#include <utility>

#include "src/core/npu/npu.hpp"
#include "src/core/platform/library.hpp"

namespace gufo::npu {
namespace {

std::string ReadLine(const std::filesystem::path& path) {
  std::ifstream in(path);
  std::string line;
  if (in)
    std::getline(in, line);
  while (!line.empty() &&
         (line.back() == '\n' || line.back() == '\r' || line.back() == ' '))
    line.pop_back();
  return line;
}

// "0x1022" -> "1022"
std::string PciHex(std::string value) {
  if (value.rfind("0x", 0) == 0)
    value.erase(0, 2);
  return value;
}

}  // namespace

std::vector<DeviceInfo> ProbeDevices(const std::filesystem::path& sysfs_root) {
  std::vector<DeviceInfo> out;
  std::error_code ec;
  const std::filesystem::path accel = sysfs_root / "class" / "accel";
  for (const auto& entry : std::filesystem::directory_iterator(accel, ec)) {
    const std::filesystem::path device = entry.path() / "device";
    const std::filesystem::path driver =
        std::filesystem::read_symlink(device / "driver", ec);
    if (ec) {
      ec.clear();
      continue;
    }
    DeviceInfo info;
    info.node = entry.path().filename().string();
    info.driver = driver.filename().string();
    if (info.driver != "amdxdna")
      continue;
    info.pci_id = PciHex(ReadLine(device / "vendor")) + ":" +
                  PciHex(ReadLine(device / "device"));
    info.firmware = ReadLine(device / "fw_version");
    out.push_back(std::move(info));
  }
  std::sort(
      out.begin(), out.end(),
      [](const DeviceInfo& a, const DeviceInfo& b) { return a.node < b.node; });
  return out;
}

void Instructions::Assign(std::span<const std::uint32_t> words) {
  if (words.size() > capacity()) {
    throw Error("instruction stream of " + std::to_string(words.size()) +
                " words exceeds its buffer of " + std::to_string(capacity()));
  }
  std::memcpy(buffer_.data(), words.data(), words.size_bytes());
  count_ = words.size();
  buffer_.ToDevice(0, words.size_bytes());
}

void Instructions::Apply(std::span<const Patch> patches, std::uint32_t value) {
  if (patches.empty())
    return;
  const std::span<std::uint32_t> w = words();
  std::size_t lo = w.size(), hi = 0;
  for (const Patch& p : patches) {
    if (p.index >= w.size())
      throw Error("instruction patch beyond the stream");
    w[p.index] += value * p.delta;
    lo = std::min<std::size_t>(lo, p.index);
    hi = std::max<std::size_t>(hi, p.index + 1);
  }
  buffer_.ToDevice(lo * 4, (hi - lo) * 4);
}

std::vector<Instructions::Patch> DiffInstructions(
    std::span<const std::uint32_t> at0, std::span<const std::uint32_t> at1) {
  if (at0.size() != at1.size())
    throw Error("instruction streams differ in length");
  std::vector<Instructions::Patch> out;
  for (std::size_t i = 0; i < at0.size(); ++i) {
    if (at0[i] != at1[i])
      out.push_back({static_cast<std::uint32_t>(i), at1[i] - at0[i]});
  }
  return out;
}

struct DispatchLibrary::Impl {
  explicit Impl(const std::filesystem::path& path) : library(path.string()) {}
  platform::Library library;
  void* symbol{};
  std::size_t scalars{};
};

DispatchLibrary::DispatchLibrary() = default;
DispatchLibrary::~DispatchLibrary() = default;
DispatchLibrary::DispatchLibrary(DispatchLibrary&&) noexcept = default;
DispatchLibrary& DispatchLibrary::operator=(DispatchLibrary&&) noexcept =
    default;

DispatchLibrary DispatchLibrary::Open(const std::filesystem::path& library,
                                      std::size_t scalars, const char* symbol) {
  if (scalars > kMaxScalars)
    throw Error("dispatch libraries take at most 16 scalars");
  DispatchLibrary out;
  try {
    // Windows loads with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, which rejects
    // relative paths.
    out.impl_ = std::make_unique<Impl>(std::filesystem::absolute(library));
    out.impl_->symbol = out.impl_->library.Symbol(symbol);
  } catch (const std::exception& e) {
    throw Error(library.string() + ": " + e.what());
  }
  out.impl_->scalars = scalars;
  return out;
}

namespace {

// dispatch_generate(s0, ..., s{N-1}, uint32_t** out): one signature per arity.
template<std::size_t... I>
std::int64_t CallWith(void* symbol, const std::int32_t* s, std::uint32_t** out,
                      std::index_sequence<I...>) {
  using Fn = std::int64_t (*)(decltype(I, std::int32_t{})..., std::uint32_t**);
  return reinterpret_cast<Fn>(symbol)(s[I]..., out);
}

template<std::size_t N>
std::int64_t CallN(void* symbol, const std::int32_t* s, std::uint32_t** out) {
  return CallWith(symbol, s, out, std::make_index_sequence<N>{});
}

template<std::size_t... N>
constexpr auto MakeTable(std::index_sequence<N...>) {
  using Call = std::int64_t (*)(void*, const std::int32_t*, std::uint32_t**);
  return std::array<Call, sizeof...(N)>{&CallN<N>...};
}

constexpr auto kCalls =
    MakeTable(std::make_index_sequence<DispatchLibrary::kMaxScalars + 1>{});

}  // namespace

std::span<const std::uint32_t> DispatchLibrary::Generate(
    std::span<const std::int32_t> scalars) const {
  if (!impl_)
    throw Error("dispatch library is not open");
  if (scalars.size() != impl_->scalars)
    throw Error("dispatch library scalar count mismatch");
  std::uint32_t* words = nullptr;
  const std::int64_t n =
      kCalls[impl_->scalars](impl_->symbol, scalars.data(), &words);
  if (n <= 0 || words == nullptr)
    throw Error("dispatch library generated no instructions");
  return {words, static_cast<std::size_t>(n)};
}

}  // namespace gufo::npu
