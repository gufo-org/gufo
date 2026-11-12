// GPU parity test: the MiniMax H3 VisualVAE *encoder* port against the
// oracle fixture captured by tools/h3/gufo-h3-vae-encoder-oracle.py from the
// checkpoint's shipped bundle (cross-checked against diffusers). Skips with
// 77 when the external checkpoint or fixture is absent.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/json.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/sha256.hpp"
#include "src/models/minimax_h3/video_vae.hpp"
#include "src/models/minimax_h3/video_vae_encoder.hpp"

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

struct Bounds {
  double relative_l2{0.05};
  double relative_max{0.05};
};

void Compare(const std::vector<float>& expected,
             const std::vector<float>& actual, const Bounds& bounds,
             const std::string& label) {
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
      reference > 0.0 ? std::sqrt(difference / reference) : std::sqrt(
                                                              difference);
  const double scale =
      reference > 0.0 ? std::sqrt(reference /
                                  static_cast<double>(expected.size()))
                      : 1.0;
  const double relative_max = maximum / std::max(scale, 1.0e-12);
  std::cout << label << " relative_l2=" << relative_l2
            << " relative_max=" << relative_max << '\n';
  CHECK(relative_l2 <= bounds.relative_l2);
  CHECK(relative_max <= bounds.relative_max);
}

h3::json::Value ExpectedEntry(const h3::json::Value& files,
                              const std::string& name) {
  const h3::json::Value* entry = files.Find(name);
  if (entry == nullptr) {
    throw h3::json::Error("fixture lacks file: " + name);
  }
  return *entry;
}

std::size_t ElementsOf(const h3::json::Value& entry) {
  std::size_t elements = 1;
  for (const h3::json::Value& dimension : RequireField(entry, "shape").
                                             AsArray()) {
    elements *= static_cast<std::size_t>(dimension.AsInt64());
  }
  return elements;
}

std::vector<float> LoadStage(const std::filesystem::path& directory,
                             const h3::json::Value& files,
                             const std::string& name) {
  const h3::json::Value entry = ExpectedEntry(files, name);
  const std::string digest = h3::Sha256File(directory / name);
  CHECK(digest == RequireField(entry, "sha256").AsString());
  return ReadF32(directory / name, ElementsOf(entry));
}

const h3::VideoVaeEncoderStage* FindStage(
    const h3::VideoVaeEncoderTrace& trace, const std::string& name) {
  for (const h3::VideoVaeEncoderStage& stage : trace.stages) {
    if (stage.name == name) {
      return &stage;
    }
  }
  return nullptr;
}

}  // namespace

int main() {
  const char* root = std::getenv("GUFO_H3_MODEL_ROOT");
  if (root == nullptr || !std::filesystem::exists(root)) {
    std::cout << "SKIP: GUFO_H3_MODEL_ROOT unset or missing\n";
    return 77;
  }
  const std::filesystem::path fixture_path(GUFO_H3_VAE_ENCODER_FIXTURE);
  if (!std::filesystem::exists(fixture_path)) {
    std::cout << "SKIP: fixture " << fixture_path << " missing\n";
    return 77;
  }
  Bounds bounds;
  std::vector<float> input;
  std::vector<std::pair<std::string, std::vector<float>>> stages;
  std::vector<float> canvas_moments;
  std::vector<float> tile_moments;
  try {
    const h3::json::Value fixture =
        h3::json::ParseFile(fixture_path, 1U << 20U);
    CHECK(RequireField(fixture, "schema").AsString() ==
          "gufo.minimax-h3-video-vae-encoder-golden.v1");
    const h3::json::Value quality = RequireField(fixture, "quality_bounds");
    bounds.relative_l2 =
        RequireField(quality, "relative_l2").AsDouble();
    bounds.relative_max =
        RequireField(quality, "relative_max").AsDouble();
    const h3::json::Value files = RequireField(fixture, "files");
    const std::filesystem::path directory = fixture_path.parent_path();
    input = LoadStage(directory, files,
                      "video-vae-encoder-keyframe-512.f32");
    canvas_moments = LoadStage(directory, files,
                               "video-vae-encoder-keyframe-512-moments.f32");
    tile_moments = LoadStage(directory, files,
                             "video-vae-encoder-tile-00-moments.f32");
    for (const std::string name : {"video-vae-encoder-stage-conv_in.f32",
                                   "video-vae-encoder-stage-down-0.f32",
                                   "video-vae-encoder-stage-down-1.f32",
                                   "video-vae-encoder-stage-down-2.f32",
                                   "video-vae-encoder-stage-down-3.f32",
                                   "video-vae-encoder-stage-down-4.f32",
                                   "video-vae-encoder-stage-down-5.f32",
                                   "video-vae-encoder-stage-norm_out.f32",
                                   "video-vae-encoder-stage-pre_quant.f32",
                                   "video-vae-encoder-stage-moments.f32"}) {
      stages.emplace_back(name, LoadStage(directory, files, name));
    }
  } catch (const std::exception& exception) {
    std::cerr << "FAIL fixture load: " << exception.what() << '\n';
    return 1;
  }
  if (failures != 0) {
    return 1;
  }

  std::string error;
  const auto inventory = h3::ModelInventory::Inspect(
      std::filesystem::path(root), GUFO_H3_SOURCE_MANIFEST, &error);
  if (!inventory.has_value()) {
    std::cerr << "FAIL inventory: " << error << '\n';
    return 1;
  }
  auto encoder = h3::VideoVaeEncoder::Create(*inventory, nullptr, nullptr,
                                             &error);
  if (encoder == nullptr) {
    std::cerr << "FAIL encoder create: " << error << '\n';
    return 1;
  }

  // Full 512x512 tiled encode (9 tiles with latent-aligned overlaps).
  std::vector<float> moments;
  h3::VideoVaeEncoderTrace canvas_trace;
  if (!encoder->EncodeKeyframeMoments(input.data(), 512, 512, nullptr,
                                      &moments, &canvas_trace, nullptr,
                                      nullptr, nullptr, &error)) {
    std::cerr << "FAIL canvas encode: " << error << '\n';
    return 1;
  }
  Compare(canvas_moments, moments, bounds, "canvas.moments");
  const std::string tile_key = "video-vae-encoder-tile-00-moments.f32";
  Compare(tile_moments, canvas_trace.Find("moments") != nullptr
                            ? canvas_trace.Find("moments")->values
                            : std::vector<float>{},
          bounds, "tile00.moments");

  // Single-tile 64x64 encode: every stage activation in isolation.
  std::vector<float> stage_input;
  stage_input.assign(3U * 64U * 64U, 0.0F);
  for (std::size_t channel = 0; channel < 3U; ++channel) {
    for (std::size_t y = 0; y < 64U; ++y) {
      for (std::size_t x = 0; x < 64U; ++x) {
        stage_input[(channel * 64U + y) * 64U + x] =
            input[(channel * 512U + y) * 512U + x];
      }
    }
  }
  std::vector<float> stage_moments;
  h3::VideoVaeEncoderTrace trace;
  if (!encoder->EncodeKeyframeMoments(stage_input.data(), 64, 64, nullptr,
                                      &stage_moments, &trace, nullptr,
                                      nullptr, nullptr, &error)) {
    std::cerr << "FAIL stage encode: " << error << '\n';
    return 1;
  }
  const std::size_t expected_stages =
      1U /* conv_in */ + 6U /* down levels */ + 1U /* norm_out */ +
      1U /* pre_quant */ + 1U /* moments */;
  CHECK(trace.stages.size() == expected_stages);
  const std::pair<const char*, const char*> stage_names[] = {
      {"conv_in", "video-vae-encoder-stage-conv_in.f32"},
      {"down.0", "video-vae-encoder-stage-down-0.f32"},
      {"down.1", "video-vae-encoder-stage-down-1.f32"},
      {"down.2", "video-vae-encoder-stage-down-2.f32"},
      {"down.3", "video-vae-encoder-stage-down-3.f32"},
      {"down.4", "video-vae-encoder-stage-down-4.f32"},
      {"down.5", "video-vae-encoder-stage-down-5.f32"},
      {"norm_out", "video-vae-encoder-stage-norm_out.f32"},
      {"pre_quant", "video-vae-encoder-stage-pre_quant.f32"},
      {"moments", "video-vae-encoder-stage-moments.f32"},
  };
  for (std::size_t index = 0; index < std::size(stage_names); ++index) {
    const h3::VideoVaeEncoderStage* stage = FindStage(trace, stage_names[index].first);
    if (stage == nullptr) {
      std::cerr << "FAIL missing stage " << stage_names[index].first << '\n';
      ++failures;
      continue;
    }
    Compare(stages[index].second, stage->values, bounds,
            stage_names[index].first);
  }

  if (failures != 0) {
    std::cerr << failures << " MiniMax H3 VAE encoder checks failed\n";
    return 1;
  }
  std::cout << "MiniMax H3 VAE encoder parity checks passed\n";
  return 0;
}
