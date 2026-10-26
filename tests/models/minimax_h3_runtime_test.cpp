#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/models/minimax_h3/json.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/sha256.hpp"
#include "tests/support/temporary_directory.hpp"

namespace {

using gufo::minimax_h3::CancellationToken;
using gufo::minimax_h3::FailureInjector;
using gufo::minimax_h3::InspectionOptions;
using gufo::minimax_h3::LoadOptions;
using gufo::minimax_h3::ModelInventory;
using gufo::minimax_h3::Phase;
using gufo::minimax_h3::PhaseSession;
using gufo::minimax_h3::ResidencyBackend;
using gufo::minimax_h3::ResidencyMode;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

struct TensorSpec {
  std::string name;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::uint64_t begin{0};
  std::uint64_t end{0};
};

std::uint64_t WriteSafetensors(const std::filesystem::path& path,
                               std::vector<TensorSpec>* tensors) {
  std::filesystem::create_directories(path.parent_path());
  std::ostringstream header;
  header << '{';
  std::uint64_t offset = 0;
  for (std::size_t index = 0; index < tensors->size(); ++index) {
    TensorSpec& tensor = (*tensors)[index];
    std::uint64_t elements = 1;
    for (std::uint64_t dimension : tensor.shape) {
      elements *= dimension;
    }
    const std::uint64_t element_bytes = tensor.dtype == "BF16" ? 2 : 4;
    tensor.begin = offset;
    tensor.end = offset + elements * element_bytes;
    offset = tensor.end;
    if (index != 0) {
      header << ',';
    }
    header << '"' << tensor.name << "\":{\"dtype\":\"" << tensor.dtype
           << "\",\"shape\":[";
    for (std::size_t dimension = 0; dimension < tensor.shape.size();
         ++dimension) {
      if (dimension != 0) {
        header << ',';
      }
      header << tensor.shape[dimension];
    }
    header << "],\"data_offsets\":[" << tensor.begin << ',' << tensor.end
           << "]}";
  }
  header << '}';
  const std::string header_bytes = header.str();
  std::ofstream output(path, std::ios::binary);
  const std::uint64_t length = header_bytes.size();
  for (int byte = 0; byte < 8; ++byte) {
    output.put(static_cast<char>((length >> (byte * 8)) & 0xFFU));
  }
  output.write(header_bytes.data(),
               static_cast<std::streamsize>(header_bytes.size()));
  for (std::uint64_t index = 0; index < offset; ++index) {
    output.put(static_cast<char>((index * 17U + 3U) & 0xFFU));
  }
  output.close();
  return std::filesystem::file_size(path);
}

void WriteText(const std::filesystem::path& path, const std::string& value) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output << value;
}

struct FixtureTensor {
  std::string component;
  std::filesystem::path shard;
  TensorSpec tensor;
};

struct ModelFixture {
  explicit ModelFixture(const std::filesystem::path& root) : root(root) {
    text = {{"model.language_model.embed_tokens.weight", "BF16", {4}}};
    transformer = {
        {"blocks.0.adaln_proj.linear.weight", "BF16", {4}},
        {"blocks.0.attn.qkv_proj.weight", "BF16", {8}},
    };
    video = {{"decoder.proj_out.weight", "F32", {4}}};
    audio = {{"decoder.weight", "F32", {4}}};
    AddShard("text_encoder",
             "FL2VA/text_encoder/model-00001-of-00001.safetensors", &text);
    AddShard("transformer",
             "FL2VA/transformer/model-00001-of-00001.safetensors",
             &transformer);
    AddShard("video_vae", "FL2VA/video_vae/source/model.safetensors", &video);
    AddShard("audio_vae", "FL2VA/audio_vae/model.safetensors", &audio);
    WriteIndexes(false);
    WriteManifest();
  }

  void AddShard(const std::string& component,
                const std::filesystem::path& relative,
                std::vector<TensorSpec>* specs) {
    const std::uint64_t size = WriteSafetensors(root / relative, specs);
    shard_sizes[relative] = size;
    for (const auto& tensor : *specs) {
      tensors.push_back({component, relative, tensor});
    }
  }

  void WriteIndexes(bool extra_transformer) const {
    WriteText(root / "FL2VA/text_encoder/model.safetensors.index.json",
              "{\"weight_map\":{\"model.language_model.embed_tokens.weight\":"
              "\"model-00001-of-00001.safetensors\"}}");
    std::ostringstream transformer_index;
    transformer_index << "{\"weight_map\":{"
                      << "\"blocks.0.adaln_proj.linear.weight\":"
                         "\"model-00001-of-00001.safetensors\","
                      << "\"blocks.0.attn.qkv_proj.weight\":"
                         "\"model-00001-of-00001.safetensors\"";
    if (extra_transformer) {
      transformer_index << ",\"blocks.99.unknown.weight\":"
                           "\"model-00001-of-00001.safetensors\"";
    }
    transformer_index << "}}";
    WriteText(root / "FL2VA/transformer/model.safetensors.index.json",
              transformer_index.str());
  }

  void WriteManifest() const {
    std::ostringstream manifest;
    manifest << "{\"schema\":\"gufo.minimax-h3-source.v1\","
             << "\"model_kind\":\"minimax-h3-fl2va-bf16\","
             << "\"revision\":\"42ed227ee7df40d41602854ae760620d6eb651fe\","
             << "\"files\":[";
    bool first = true;
    for (const auto& [path, size] : shard_sizes) {
      if (!first) {
        manifest << ',';
      }
      first = false;
      manifest << "{\"path\":\"" << path.generic_string()
               << "\",\"size\":" << size << '}';
    }
    manifest << "],\"tensors\":[";
    first = true;
    for (const auto& entry : tensors) {
      if (!first) {
        manifest << ',';
      }
      first = false;
      manifest << "{\"component\":\"" << entry.component << "\","
               << "\"name\":\"" << entry.tensor.name << "\","
               << "\"dtype\":\"" << entry.tensor.dtype << "\","
               << "\"shape\":[";
      for (std::size_t index = 0; index < entry.tensor.shape.size(); ++index) {
        if (index != 0) {
          manifest << ',';
        }
        manifest << entry.tensor.shape[index];
      }
      manifest << "],\"shard\":\"" << entry.shard.generic_string()
               << "\",\"data_offsets\":[" << entry.tensor.begin << ','
               << entry.tensor.end
               << "],\"byte_count\":" << (entry.tensor.end - entry.tensor.begin)
               << '}';
    }
    manifest << "]}";
    WriteText(root / "source-manifest.json", manifest.str());
  }

  std::filesystem::path root;
  std::vector<TensorSpec> text;
  std::vector<TensorSpec> transformer;
  std::vector<TensorSpec> video;
  std::vector<TensorSpec> audio;
  std::map<std::filesystem::path, std::uint64_t> shard_sizes;
  std::vector<FixtureTensor> tensors;
};

class FakeBackend final : public ResidencyBackend {
public:
  ~FakeBackend() override {
    Expect(allocations.empty(), "fake backend allocations leaked");
    Expect(registrations.empty(), "fake backend registrations leaked");
    Expect(streams.empty(), "fake backend streams leaked");
  }

  bool IsSupportedTarget(std::string*) override { return true; }

  bool CreateStream(void** stream, std::string*) override {
    auto value = std::make_unique<int>(1);
    *stream = value.get();
    streams.emplace(*stream, std::move(value));
    return true;
  }

  void DestroyStream(void* stream) noexcept override { streams.erase(stream); }

  bool Allocate(std::size_t bytes, void** device, std::string*) override {
    auto value = std::make_unique<std::byte[]>(std::max<std::size_t>(bytes, 1));
    *device = value.get();
    allocations.emplace(*device, Allocation{bytes, std::move(value)});
    return true;
  }

  void Free(void* device) noexcept override { allocations.erase(device); }

  bool RegisterReadOnlyMapped(void* host, std::size_t bytes,
                              void** device_alias, std::string*) override {
    registrations.emplace(host, bytes);
    *device_alias = host;
    return true;
  }

  void Unregister(void* host) noexcept override { registrations.erase(host); }

  bool CopyToDevice(void* device, const void* host, std::size_t bytes, void*,
                    std::string*) override {
    std::memcpy(device, host, bytes);
    copied_bytes += bytes;
    return true;
  }

  bool Synchronize(void*, std::string*) override { return true; }

  [[nodiscard]] std::size_t LiveResources() const {
    return allocations.size() + registrations.size() + streams.size();
  }

  struct Allocation {
    std::size_t bytes;
    std::unique_ptr<std::byte[]> data;
  };
  std::unordered_map<void*, Allocation> allocations;
  std::unordered_map<void*, std::size_t> registrations;
  std::unordered_map<void*, std::unique_ptr<int>> streams;
  std::size_t copied_bytes{0};
};

ModelInventory InspectFixture(const ModelFixture& fixture) {
  std::string error;
  auto inventory = ModelInventory::Inspect(
      fixture.root, fixture.root / "source-manifest.json",
      InspectionOptions{false}, &error);
  Expect(inventory.has_value(), error);
  return std::move(*inventory);
}

void TestStrictJson() {
  using gufo::minimax_h3::json::Error;
  using gufo::minimax_h3::json::Parse;
  const auto parsed = Parse(
      "{\"u\":18446744073709551615,\"s\":-7,\"f\":1e-6,"
      "\"unicode\":\"\\uD83E\\uDD8A\"}");
  Expect(parsed.Find("u")->AsUint64() == UINT64_MAX, "JSON preserves uint64");
  Expect(parsed.Find("s")->AsInt64() == -7, "JSON preserves int64");
  Expect(parsed.Find("unicode")->AsString() == "\xF0\x9F\xA6\x8A",
         "JSON decodes surrogate pairs");
  bool duplicate_rejected = false;
  try {
    (void)Parse("{\"x\":1,\"x\":2}");
  } catch (const Error&) {
    duplicate_rejected = true;
  }
  Expect(duplicate_rejected, "JSON duplicate keys rejected");
  const std::string abc = "abc";
  Expect(gufo::minimax_h3::Sha256(std::span(
             reinterpret_cast<const unsigned char*>(abc.data()), abc.size())) ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "SHA-256 known-answer test");
}

void TestInspectionAndInventory() {
  TemporaryDirectory temporary("gufo-h3-runtime");
  ModelFixture fixture(temporary.path());
  const ModelInventory inventory = InspectFixture(fixture);
  Expect(inventory.tensors().size() == 5, "all fixture tensors inspected");
  Expect(inventory.shards().size() == 4, "all fixture shards inspected");
  Expect(inventory.telemetry().payload_bytes_read == 0,
         "metadata inspection reads no tensor payload");
  Expect(inventory.telemetry().mapped_bytes == 0,
         "metadata inspection maps no weights");
  Expect(inventory.telemetry().device_bytes == 0,
         "metadata inspection allocates no device memory");
  Expect(inventory.TensorsForPhase(Phase::kAdaLnPrecompute).size() == 1,
         "AdaLN tensor classified separately");
  Expect(inventory.TensorsForPhase(Phase::kDitCore).size() == 1,
         "DiT core excludes AdaLN tensor");
}

void TestUnknownArchitectureRejected() {
  TemporaryDirectory temporary("gufo-h3-runtime");
  ModelFixture fixture(temporary.path());
  fixture.WriteIndexes(true);
  std::string error;
  const auto inventory = ModelInventory::Inspect(
      fixture.root, fixture.root / "source-manifest.json",
      InspectionOptions{false}, &error);
  Expect(!inventory.has_value(), "unknown indexed tensor rejected");
  Expect(error.find("unknown tensor") != std::string::npos,
         "unknown tensor diagnostic retained");
}

void TestPinnedManifestRequiredByDefault() {
  TemporaryDirectory temporary("gufo-h3-runtime");
  ModelFixture fixture(temporary.path());
  std::string error;
  const auto inventory = ModelInventory::Inspect(
      fixture.root, fixture.root / "source-manifest.json", &error);
  Expect(!inventory.has_value(), "look-alike source manifest rejected");
  Expect(error.find("SHA-256") != std::string::npos,
         "pinned-manifest rejection identifies SHA-256");
  Expect(gufo::minimax_h3::DefaultResidencyMode() == ResidencyMode::kDeviceCopy,
         "measured initial default is explicit device copy");
}

void TestDeviceCopyAndMappedResidency() {
  TemporaryDirectory temporary("gufo-h3-runtime");
  ModelFixture fixture(temporary.path());
  const ModelInventory inventory = InspectFixture(fixture);
  FakeBackend backend;
  {
    LoadOptions options;
    options.mode = ResidencyMode::kDeviceCopy;
    options.persistent_bytes = 128;
    options.scratch_bytes = 256;
    std::string error;
    auto session = PhaseSession::Load(inventory, Phase::kDitCore, options,
                                      backend, nullptr, nullptr, &error);
    Expect(session.has_value(), error);
    Expect(session->tensors().size() == 1, "DiT core loaded one tensor");
    Expect(session->telemetry().device_weight_bytes >= 16,
           "device copy accounts weight bytes");
    Expect(session->telemetry().registered_host_bytes == 0,
           "device copy releases file mappings");
    const std::size_t resources_before = backend.LiveResources();
    session->BeginTimedExecution();
    Expect(backend.LiveResources() == resources_before,
           "timed transition allocates nothing");
  }
  Expect(backend.LiveResources() == 0, "device-copy phase fully released");

  {
    LoadOptions options;
    options.mode = ResidencyMode::kMappedReadOnly;
    options.persistent_bytes = 64;
    options.scratch_bytes = 96;
    std::string error;
    auto session = PhaseSession::Load(inventory, Phase::kAudioVae, options,
                                      backend, nullptr, nullptr, &error);
    Expect(session.has_value(), error);
    Expect(session->telemetry().registered_host_bytes > 0,
           "mapped mode accounts registered bytes");
    Expect(session->telemetry().device_weight_bytes == 0,
           "mapped mode does not copy weights");
  }
  Expect(backend.LiveResources() == 0, "mapped phase fully released");
}

void TestFailureInjectionIsTransactional() {
  TemporaryDirectory temporary("gufo-h3-runtime");
  ModelFixture fixture(temporary.path());
  const ModelInventory inventory = InspectFixture(fixture);
  for (const ResidencyMode mode :
       {ResidencyMode::kMappedReadOnly, ResidencyMode::kDeviceCopy}) {
    LoadOptions options;
    options.mode = mode;
    options.persistent_bytes = 128;
    options.scratch_bytes = 256;

    std::size_t operation_count = 0;
    {
      FakeBackend backend;
      FailureInjector observer;
      std::string error;
      auto session =
          PhaseSession::Load(inventory, Phase::kPromptEncoder, options, backend,
                             nullptr, &observer, &error);
      Expect(session.has_value(), error);
      operation_count = observer.operations();
    }
    Expect(operation_count >= 7, "failure test observes every load stage");

    for (std::size_t fail_after = 0; fail_after < operation_count;
         ++fail_after) {
      FakeBackend backend;
      FailureInjector failure(fail_after);
      std::string error;
      auto session =
          PhaseSession::Load(inventory, Phase::kPromptEncoder, options, backend,
                             nullptr, &failure, &error);
      Expect(!session.has_value(),
             "injected failure must reject " +
                 std::string(gufo::minimax_h3::ToString(mode)) +
                 " at operation " + std::to_string(fail_after));
      Expect(backend.LiveResources() == 0,
             "injected failure restores resource baseline");
    }
  }
}

void TestCancellationAndPhasePeaks() {
  TemporaryDirectory temporary("gufo-h3-runtime");
  ModelFixture fixture(temporary.path());
  const ModelInventory inventory = InspectFixture(fixture);
  FakeBackend backend;
  CancellationToken cancelled;
  cancelled.Cancel();
  std::string error;
  auto rejected =
      PhaseSession::Load(inventory, Phase::kVisualVae, LoadOptions{}, backend,
                         &cancelled, nullptr, &error);
  Expect(!rejected.has_value(), "pre-cancelled phase rejected");
  Expect(backend.LiveResources() == 0, "cancelled phase leaks no resources");

  std::uint64_t prompt_peak = 0;
  {
    auto prompt = PhaseSession::Load(
        inventory, Phase::kPromptEncoder,
        LoadOptions{ResidencyMode::kDeviceCopy, 64, 128, true}, backend,
        nullptr, nullptr, &error);
    Expect(prompt.has_value(), error);
    prompt_peak = prompt->telemetry().peak_live_bytes;
    auto overlapping = PhaseSession::Load(
        inventory, Phase::kVisualVae,
        LoadOptions{ResidencyMode::kDeviceCopy, 32, 64, true}, backend, nullptr,
        nullptr, &error);
    Expect(!overlapping.has_value(), "overlapping phase residency rejected");
    Expect(error.find("active residency phase") != std::string::npos,
           "overlapping phase has a precise diagnostic");
  }
  Expect(backend.LiveResources() == 0, "prompt phase releases before decoder");
  std::uint64_t decoder_peak = 0;
  {
    auto decoder = PhaseSession::Load(
        inventory, Phase::kVisualVae,
        LoadOptions{ResidencyMode::kDeviceCopy, 32, 64, true}, backend, nullptr,
        nullptr, &error);
    Expect(decoder.has_value(), error);
    decoder_peak = decoder->telemetry().peak_live_bytes;
  }
  Expect(backend.LiveResources() == 0, "decoder phase releases all resources");
  Expect(prompt_peak > 0 && decoder_peak > 0,
         "phase residency reports independent peaks");
}

}  // namespace

int main() {
  TestStrictJson();
  TestInspectionAndInventory();
  TestUnknownArchitectureRejected();
  TestPinnedManifestRequiredByDefault();
  TestDeviceCopyAndMappedResidency();
  TestFailureInjectionIsTransactional();
  TestCancellationAndPhasePeaks();
  std::cout << "MiniMax H3 runtime foundation tests passed.\n";
  return 0;
}
