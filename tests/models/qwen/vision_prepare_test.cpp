// `gufo serve llm` uploads the vision sidecar while loading through
// Encoder::Upload: the weights upload once, ResidentBytes accounts for exactly
// them, and a later Encode reuses them instead of uploading again.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/core/image.hpp"
#include "src/families/qwen/vision/encoder.hpp"

namespace {
bool observe_upload{};
bool pageable_upload{};
std::size_t uploaded_bytes{};
}  // namespace

extern "C" hipError_t __real_hipMemcpyAsync(void*, const void*, std::size_t,
                                            hipMemcpyKind, hipStream_t);
extern "C" hipError_t __wrap_hipMemcpyAsync(void* destination,
                                            const void* source,
                                            std::size_t bytes,
                                            hipMemcpyKind kind,
                                            hipStream_t stream) {
  if (observe_upload && kind == hipMemcpyHostToDevice) {
    hipPointerAttribute_t attributes{};
    const auto status = hipPointerGetAttributes(&attributes, source);
    pageable_upload |=
        status != hipSuccess || attributes.type != hipMemoryTypeHost;
    if (status != hipSuccess)
      (void)hipGetLastError();
    uploaded_bytes += bytes;
  }
  return __real_hipMemcpyAsync(destination, source, bytes, kind, stream);
}

namespace {
using gufo::models::qwen::vision::Encoder;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

// On Strix Halo the device heap is system memory, so the free figure also
// moves with unrelated host allocations; compare it within this slack.
constexpr std::size_t kSlackBytes = std::size_t{64} << 20;

std::size_t FreeDeviceBytes() {
  std::size_t free = 0, total = 0;
  Require(hipMemGetInfo(&free, &total) == hipSuccess, "hipMemGetInfo failed");
  return free;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "usage: qwen_vision_prepare_test MMPROJ_BF16_GGUF\n";
    return 77;
  }
  try {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
    Require(reader != nullptr, error);
    // Every tensor except the two F32 patch embeddings, which upload as one
    // converted BF16 tensor.
    std::size_t expected = std::size_t{1152} * 3 * 2 * 16 * 16 * 2;
    for (const auto& tensor : reader->GetTensors()) {
      if (tensor.name != "v.patch_embd.weight" &&
          tensor.name != "v.patch_embd.weight.1") {
        expected += tensor.ElementCount() *
                    (tensor.type == gufo::core::GgmlType::kF32 ? 4 : 2);
      }
    }
    const auto width = static_cast<std::uint32_t>(
        reader->GetMetadataUint64("clip.vision.projection_dim").value_or(0));
    auto encoder = Encoder::Open({}, argv[1], width);
    Require(encoder != nullptr && encoder->ResidentBytes() == 0,
            "opening the sidecar uploaded weights");

    const auto before = FreeDeviceBytes();
    observe_upload = true;
    encoder->Upload();
    observe_upload = false;
    Require(!pageable_upload && uploaded_bytes == expected,
            "vision upload registered pageable source tensors with the GPU");
    const auto prepared = FreeDeviceBytes();
    Require(encoder->ResidentBytes() == expected,
            "resident bytes differ from the sidecar weights");
    Require(before > prepared && before - prepared + kSlackBytes >= expected,
            "Upload did not allocate the weights");

    encoder->Upload();
    Require(encoder->ResidentBytes() == expected &&
                FreeDeviceBytes() + kSlackBytes >= prepared,
            "a second Upload allocated the weights again");

    gufo::core::Image image{
        .width = 256,
        .height = 256,
        .pixels = std::vector<std::uint8_t>(std::size_t{256} * 256 * 3, 128)};
    {
      observe_upload = true;
      const auto embedding = encoder->Encode(image);
      observe_upload = false;
      Require(
          !pageable_upload && uploaded_bytes == expected + image.pixels.size(),
          "vision input upload registered pageable pixels with the GPU");
      Require(embedding->rows() == 64 && embedding->width() == width,
              "unexpected embedding shape");
    }
    const auto encoded = FreeDeviceBytes();
    Require(encoder->ResidentBytes() == expected &&
                encoded + kSlackBytes >= prepared,
            "Encode uploaded or retained weights after Upload");
    std::cout << "vision prepare resident_bytes=" << expected
              << " upload_free_delta=" << before - prepared
              << " encode_free_delta="
              << static_cast<long long>(prepared) -
                     static_cast<long long>(encoded)
              << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "qwen_vision_prepare_test: " << e.what() << '\n';
    return 1;
  }
}
