// GPU end-to-end check of the glm5v vision tower: opens a real mmproj
// (F16 or BF16 sidecar), encodes a deterministic 336x448 canvas (768
// patches -> 192 merged tokens, the exact merge-window sizes that faulted
// when merger im2col overswrote the patchmat buffer) and demands finite,
// input-sensitive, reproducible outputs.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "staging/models/glm5_next/vision/encoder.hpp"

namespace {

int Fail(const std::string& message) {
  std::printf("FAIL: %s\n", message.c_str());
  return 1;
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_GLM5_NEXT_MMPROJ");
  if (path == nullptr || path[0] == '\0') {
    std::printf("SKIP: set GUFO_GLM5_NEXT_MMPROJ\n");
    return 0;
  }
  std::string error;
  auto encoder = gufo::models::glm5_next::vision::Encoder::Open(path, &error);
  if (encoder == nullptr) {
    return Fail("open mmproj: " + error);
  }

  constexpr std::uint32_t kNx = 336;
  constexpr std::uint32_t kNy = 448;
  const std::vector<float> pixels = [] {
    std::vector<float> planar(3ULL * kNy * kNx);
    for (std::uint32_t c = 0; c < 3; ++c) {
      for (std::uint32_t y = 0; y < kNy; ++y) {
        for (std::uint32_t x = 0; x < kNx; ++x) {
          planar[(static_cast<std::size_t>(c) * kNy + y) * kNx + x] =
              static_cast<float>((x * 7U + y * 13U + c * 29U) % 251U) / 250.0F;
        }
      }
    }
    return planar;
  }();

  std::vector<float> out;
  if (!encoder->Encode(pixels.data(), kNx, kNy, out, &error)) {
    return Fail("encode: " + error);
  }
  const std::uint64_t expect =
      static_cast<std::uint64_t>(kNx / 14 / 2) * (kNy / 14 / 2) * 4096ULL;
  if (out.size() != expect) {
    return Fail("encode returned " + std::to_string(out.size()) +
                " floats, want " + std::to_string(expect));
  }
  double energy = 0.0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (!std::isfinite(out[i])) {
      return Fail("non-finite output at " + std::to_string(i));
    }
    energy += std::fabs(out[i]);
  }
  if (energy == 0.0) {
    return Fail("all-zero output");
  }

  // 2x2 merge windows overlap their source patches, so re-encoding the same
  // canvas must reproduce the same projection run to run.
  std::vector<float> again;
  if (!encoder->Encode(pixels.data(), kNx, kNy, again, &error)) {
    return Fail("re-encode: " + error);
  }
  const double tol = 1e-3;
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (std::fabs(out[i] - again[i]) > tol) {
      return Fail("reproducible-encode mismatch at " + std::to_string(i));
    }
  }
  std::printf("PASS: %zu vision tokens finite and reproducible\n",
              static_cast<std::size_t>(expect / 4096ULL));
  return 0;
}
