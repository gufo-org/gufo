// GPU parity test: MiniMax H3 Qwen3-VL vision tower port against the
// transformers oracle captured by tools/h3/gufo-h3-vision-oracle.py. Also
// validates the engine's patch builder (merge-block row order, row layout,
// and (x/255 - 0.5)/0.5 normalization) against the shipped processor.
// Skips with 77 when the external checkpoint or fixture is absent.

#include <cmath>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/minimax_h3/image_processor.hpp"
#include "src/models/minimax_h3/json.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/sha256.hpp"
#include "src/models/minimax_h3/vision_tower.hpp"

namespace {

namespace h3 = gufo::minimax_h3;

int failures = 0;

#define CHECK(condition)                                          \
  do {                                                            \
    if (!(condition)) {                                           \
      std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << ": " \
                << #condition << '\n';                            \
      ++failures;                                                 \
    }                                                             \
  } while (false)

std::vector<float> ReadF32(const std::filesystem::path& path,
                           std::size_t elements) {
  std::vector<float> values(elements);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(values.data()),
                  static_cast<std::streamsize>(values.size() *
                                               sizeof(float))) ||
      input.peek() != std::ifstream::traits_type::eof()) {
    values.clear();
  }
  return values;
}

const h3::json::Value& RequireField(const h3::json::Value& value,
                                    std::string_view name) {
  const h3::json::Value* found = value.Find(name);
  if (found == nullptr) {
    throw h3::json::Error("missing fixture field: " + std::string(name));
  }
  return *found;
}

std::size_t ElementsOf(const h3::json::Value& entry) {
  std::size_t elements = 1;
  for (const h3::json::Value& dimension :
       RequireField(entry, "shape").AsArray()) {
    elements *= static_cast<std::size_t>(dimension.AsInt64());
  }
  return elements;
}

std::vector<float> LoadBlob(const std::filesystem::path& directory,
                            const h3::json::Value& files,
                            const std::string& name) {
  const h3::json::Value* entry = files.Find(name);
  if (entry == nullptr) {
    throw h3::json::Error("fixture lacks file: " + name);
  }
  CHECK(h3::Sha256File(directory / name) ==
        RequireField(*entry, "sha256").AsString());
  return ReadF32(directory / name, ElementsOf(*entry));
}

void Compare(const std::vector<float>& expected,
             const std::vector<float>& actual, double relative_l2_bound,
             double relative_max_bound, const std::string& label) {
  if (expected.size() != actual.size()) {
    std::cerr << "FAIL " << label << ": size " << actual.size() << " != "
              << expected.size() << '\n';
    ++failures;
    return;
  }
  double reference = 0.0;
  double difference = 0.0;
  double maximum = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!std::isfinite(actual[i])) {
      std::cerr << "FAIL " << label << ": non-finite at " << i << '\n';
      ++failures;
      return;
    }
    reference += static_cast<double>(expected[i]) * expected[i];
    const double delta =
        static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
    difference += delta * delta;
    maximum = std::max(maximum, std::fabs(delta));
  }
  const double relative_l2 =
      reference > 0.0 ? std::sqrt(difference / reference) : 0.0;
  const double scale =
      reference > 0.0
          ? std::sqrt(reference / static_cast<double>(expected.size()))
          : 1.0;
  const double relative_max = maximum / std::max(scale, 1.0e-12);
  std::cout << label << " relative_l2=" << relative_l2
            << " relative_max=" << relative_max << '\n';
  CHECK(relative_l2 <= relative_l2_bound);
  CHECK(relative_max <= relative_max_bound);
}

}  // namespace

int main() {
  const char* root = std::getenv("GUFO_H3_MODEL_ROOT");
  if (root == nullptr || !std::filesystem::exists(root)) {
    std::cout << "SKIP: GUFO_H3_MODEL_ROOT unset or missing\n";
    return 77;
  }
  const std::filesystem::path fixture_path(GUFO_H3_VISION_FIXTURE);
  if (!std::filesystem::exists(fixture_path)) {
    std::cout << "SKIP: fixture " << fixture_path << " missing\n";
    return 77;
  }
  std::string error;
  std::vector<float> pixels_expected;
  std::vector<float> output_expected;
  std::vector<std::vector<float>> deep_expected;
  std::vector<std::uint8_t> rgb;
  int width = 0;
  int height = 0;
  double l2_bound = 0.05;
  double max_bound = 0.05;
  try {
    const h3::json::Value fixture =
        h3::json::ParseFile(fixture_path, 1U << 20U);
    CHECK(RequireField(fixture, "schema").AsString() ==
          "gufo.minimax-h3-vision-tower-golden.v1");
    const h3::json::Value quality = RequireField(fixture, "quality_bounds");
    l2_bound = RequireField(quality, "relative_l2").AsDouble();
    max_bound = RequireField(quality, "relative_max").AsDouble();
    const h3::json::Value files = RequireField(fixture, "files");
    const std::filesystem::path directory = fixture_path.parent_path();
    const h3::json::Value image = RequireField(fixture, "image");
    width = static_cast<int>(image.Find("width")->AsInt64());
    height = static_cast<int>(image.Find("height")->AsInt64());
    CHECK(h3::Sha256File(directory / RequireField(image, "path").AsString()) ==
          RequireField(image, "sha256").AsString());
    rgb = [&] {
      const std::vector<std::uint8_t> bytes =
          gufo::core::ReadImageFile(directory /
                                    RequireField(image, "path").AsString());
      return gufo::core::DecodeImage(bytes).pixels;
    }();
    pixels_expected = LoadBlob(directory, files, "video-vision-pixels.f32");
    output_expected = LoadBlob(directory, files, "video-vision-output.f32");
    for (const std::string name : {"video-vision-deep-0.f32",
                                   "video-vision-deep-1.f32",
                                   "video-vision-deep-2.f32"}) {
      deep_expected.push_back(LoadBlob(directory, files, name));
    }
  } catch (const std::exception& exception) {
    std::cerr << "FAIL fixture load: " << exception.what() << '\n';
    return 1;
  }
  if (failures != 0) {
    return 1;
  }

  // Engine patch builder must reproduce the shipped processor.
  h3::KeyframeVision vision;
  if (!h3::image_processor::BuildKeyframeVision(rgb, width, height, &vision,
                                                &error)) {
    std::cerr << "FAIL patch builder: " << error << '\n';
    return 1;
  }
  Compare(pixels_expected, vision.pixel_values, 1.0e-5, 1.0e-5,
          "processor.pixels");
  if (failures != 0) {
    return 1;
  }

  const auto inventory = h3::ModelInventory::Inspect(
      std::filesystem::path(root), GUFO_H3_SOURCE_MANIFEST, &error);
  if (!inventory.has_value()) {
    std::cerr << "FAIL inventory: " << error << '\n';
    return 1;
  }
  std::vector<std::uint16_t> embedded;
  std::vector<std::uint16_t> deep;
  if (!h3::EncodeVision(*inventory, vision, nullptr, &embedded, &deep,
                        &error)) {
    std::cerr << "FAIL EncodeVision: " << error << '\n';
    return 1;
  }
  const auto to_float = [](const std::vector<std::uint16_t>& bits) {
    std::vector<float> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
      values[i] = std::bit_cast<float>(
          static_cast<std::uint32_t>(bits[i]) << 16U);
    }
    return values;
  };
  Compare(output_expected, to_float(embedded), l2_bound, max_bound,
          "vision.merger");
  CHECK(deep.size() == 3U * embedded.size());
  if (deep.size() == 3U * embedded.size()) {
    for (std::size_t tap = 0; tap < 3U; ++tap) {
      const std::vector<std::uint16_t> slice(
          deep.begin() + static_cast<std::ptrdiff_t>(tap * embedded.size()),
          deep.begin() +
              static_cast<std::ptrdiff_t>((tap + 1U) * embedded.size()));
      Compare(deep_expected[tap], to_float(slice), l2_bound, max_bound,
              "vision.deepstack." + std::to_string(tap));
    }
  }
  if (failures != 0) {
    std::cerr << failures << " MiniMax H3 vision tower checks failed\n";
    return 1;
  }
  std::cout << "MiniMax H3 vision tower parity checks passed\n";
  return 0;
}
