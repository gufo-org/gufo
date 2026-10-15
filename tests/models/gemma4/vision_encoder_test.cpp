// HIP vision encoder against the scalar reference on a small synthetic
// image. Reads the sidecar named by GUFO_GEMMA4_MMPROJ (31B or 26B-A4B);
// skips (77) without.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/vision/encoder.hpp"
#include "src/models/gemma4/vision/reference.hpp"
#include "tests/models/gemma4/check.hpp"

namespace vision = gufo::models::gemma4::vision;
using gemma4_test::Require;

int main() {
  const char* path = std::getenv("GUFO_GEMMA4_MMPROJ");
  if (path == nullptr || *path == '\0') {
    std::cerr << "SKIP: set GUFO_GEMMA4_MMPROJ\n";
    return 77;
  }
  return gemma4_test::Run([&] {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
    Require(reader != nullptr, error);
    // The projection width of the target the sidecar belongs to: 5376 for
    // the 31B, 2816 for the 26B-A4B.
    const auto width = static_cast<std::uint32_t>(
        reader->GetMetadataUint64("clip.vision.projection_dim").value_or(0));
    const auto weights = vision::Weights::Resolve(*reader, width);
    // 144x96: 54 patches, 6 soft tokens; deterministic texture.
    gufo::core::Image image{144, 96, std::vector<std::uint8_t>(144 * 96 * 3)};
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
      image.pixels[i] = static_cast<std::uint8_t>((i * 2654435761U) >> 24);
    }
    std::map<std::string, std::vector<float>, std::less<>> expected;
    (void)vision::Reference(weights).Encode(
        image, [&](std::string_view stage, std::span<const float> data) {
          expected.emplace(stage, std::vector<float>(data.begin(), data.end()));
        });
    vision::Encoder encoder(path, width);
    std::size_t seen = 0;
    const auto embedding = encoder.Encode(
        image, [&](std::string_view stage, std::span<const float> data) {
          const auto& ref = expected.at(std::string(stage));
          Require(ref.size() == data.size(), std::string(stage) + ": size");
          double err = 0, norm = 0;
          for (std::size_t i = 0; i < data.size(); ++i) {
            err += (double{data[i]} - ref[i]) * (double{data[i]} - ref[i]);
            norm += double{ref[i]} * ref[i];
          }
          const double relative = std::sqrt(err / norm);
          // FP32 patch embedding; binary16 GEMM inputs afterwards, whose
          // rounding grows through the late layers to about 2e-3 (BF16
          // inputs reached 2e-2; docs VISION.md).
          const double limit = stage == "patch" ? 1e-5 : 1e-2;
          std::cout << stage << " rel rms " << relative << '\n';
          Require(relative < limit, std::string(stage) + " diverges");
          ++seen;
        });
    Require(seen == expected.size(), "encoder skipped stages");
    Require(embedding->rows() == 6 && embedding->width() == width,
            "embedding shape");
  });
}
