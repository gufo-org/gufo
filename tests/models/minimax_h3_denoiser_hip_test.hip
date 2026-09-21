#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/minimax_h3/denoiser.hpp"
#include "src/models/minimax_h3/dit.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/sha256.hpp"

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

std::vector<std::uint16_t> ReadBf16(const std::filesystem::path& path,
                                    std::size_t elements) {
  std::vector<std::uint16_t> values(elements);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(values.data()),
                  static_cast<std::streamsize>(values.size() *
                                               sizeof(std::uint16_t))) ||
      input.peek() != std::ifstream::traits_type::eof()) {
    return {};
  }
  return values;
}

std::vector<float> ReadF32(const std::filesystem::path& path,
                           std::size_t elements) {
  std::vector<float> values(elements);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(
          reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(float))) ||
      input.peek() != std::ifstream::traits_type::eof()) {
    return {};
  }
  return values;
}

float ToFloat(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

bool Finite(std::span<const float> values) {
  return std::ranges::all_of(values,
                             [](float value) { return std::isfinite(value); });
}

template<typename Left, typename Right, typename ConvertLeft,
         typename ConvertRight>
void CheckBoundary(std::string_view name, std::span<const Left> got,
                   std::span<const Right> expected, double relative_l2_bound,
                   double relative_max_bound, ConvertLeft convert_left,
                   ConvertRight convert_right) {
  CHECK(got.size() == expected.size());
  double error_square = 0.0;
  double value_square = 0.0;
  double maximum_error = 0.0;
  double maximum_value = 0.0;
  std::size_t first_mismatch = std::numeric_limits<std::size_t>::max();
  for (std::size_t index = 0; index < std::min(got.size(), expected.size());
       ++index) {
    const double actual = convert_left(got[index]);
    const double want = convert_right(expected[index]);
    const double delta = actual - want;
    error_square += delta * delta;
    value_square += want * want;
    maximum_error = std::max(maximum_error, std::abs(delta));
    maximum_value = std::max(maximum_value, std::abs(want));
    if (first_mismatch == std::numeric_limits<std::size_t>::max() &&
        actual != want) {
      first_mismatch = index;
    }
  }
  const double relative_l2 =
      std::sqrt(error_square / std::max(value_square, 1.0e-24));
  const double relative_max = maximum_error / std::max(maximum_value, 1.0e-12);
  std::cout << name << " relative_l2=" << relative_l2
            << " relative_max=" << relative_max
            << " first_mismatch=" << first_mismatch << '\n';
  CHECK(relative_l2 < relative_l2_bound);
  CHECK(relative_max < relative_max_bound);
}

void CheckBf16Boundary(std::string_view name,
                       std::span<const std::uint16_t> got,
                       std::span<const std::uint16_t> expected, double bound) {
  CheckBoundary(name, got, expected, bound, bound, ToFloat, ToFloat);
}

void CheckF32Boundary(std::string_view name, std::span<const float> got,
                      std::span<const float> expected, double relative_l2_bound,
                      double relative_max_bound) {
  CheckBoundary(
      name, got, expected, relative_l2_bound, relative_max_bound,
      [](float value) { return value; }, [](float value) { return value; });
}

int CheckReal() {
  const char* root = std::getenv("GUFO_H3_MODEL_ROOT");
  const char* conditioning_path = std::getenv("GUFO_H3_LAYER50_CONDITIONING");
  if (root == nullptr || conditioning_path == nullptr) {
    std::cout << "SKIP: GUFO_H3_MODEL_ROOT or "
                 "GUFO_H3_LAYER50_CONDITIONING unset\n";
    return 77;
  }
  std::string error;
  auto inventory =
      h3::ModelInventory::Inspect(root, GUFO_H3_SOURCE_MANIFEST, &error);
  CHECK(inventory.has_value());
  if (!inventory) {
    std::cerr << error << '\n';
    return 1;
  }
  constexpr std::size_t kTextRows = 6;
  auto conditioning =
      ReadBf16(conditioning_path, kTextRows * h3::kH3TextConditionWidth);
  CHECK(conditioning.size() == kTextRows * h3::kH3TextConditionWidth);
  auto geometry = h3::ResolveGenerationGeometry(32, 32, 5, &error);
  CHECK(geometry.has_value());
  if (!geometry || conditioning.empty()) {
    std::cerr << error << '\n';
    return 1;
  }
  h3::CancellationToken cancelled_creation;
  cancelled_creation.Cancel();
  h3::DenoiserTelemetry rejected_telemetry;
  auto rejected = h3::DenoiserSession::Create(
      *inventory,
      {.geometry = *geometry,
       .evaluations = 2,
       .text_rows = kTextRows,
       .active_blocks = 3},
      {.layer50 = conditioning, .tags = {}}, &cancelled_creation,
      &rejected_telemetry, &error);
  CHECK(rejected == nullptr);
  CHECK(error.find("invalid") != std::string::npos);
  error.clear();
  h3::DenoiserTelemetry creation;
  auto session = h3::DenoiserSession::Create(
      *inventory,
      {.geometry = *geometry,
       .evaluations = 2,
       .text_rows = kTextRows,
       .active_blocks = 3},
      {.layer50 = conditioning, .tags = {}}, nullptr, &creation, &error);
  CHECK(session != nullptr);
  if (session == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  CHECK(session->layout().rows() == 24);
  CHECK(session->block0_modulation().size() ==
        3 * 3 * h3::kDitModulationSlots * h3::kDitHiddenSize);
  CHECK(session->refined_text().size() == kTextRows * h3::kDitHiddenSize);
  CHECK(creation.core_weight_bytes > 0);
  CHECK(creation.core_gemm_plan_validations == 4);
  CHECK(creation.core_weight_bytes < 4ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK(creation.peak_live_bytes < 8ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK(creation.scratch_address != 0);
  CHECK(creation.swap_bytes == 0);

  const std::size_t video_elements =
      h3::kH3VideoLatentChannels *
      static_cast<std::size_t>(geometry->temporal.video_latent_frames) *
      static_cast<std::size_t>(geometry->latent_height) *
      static_cast<std::size_t>(geometry->latent_width);
  const std::size_t audio_elements =
      h3::kH3AudioLatentChannels * h3::kH3AudioTracks *
      static_cast<std::size_t>(geometry->temporal.audio_latent_frames);
  std::vector<float> video(video_elements);
  std::vector<float> audio(audio_elements);
  for (std::size_t index = 0; index < video.size(); ++index) {
    video[index] = std::sin(static_cast<float>(index) * 0.031F) * 0.5F;
  }
  for (std::size_t index = 0; index < audio.size(); ++index) {
    audio[index] = std::cos(static_cast<float>(index) * 0.019F) * 0.5F;
  }
  h3::DenoiserVelocity velocity;
  h3::DenoiserTelemetry telemetry;
  CHECK(session->Forward(0, video, audio, nullptr, &velocity, &telemetry,
                         &error));
  CHECK(Finite(velocity.video));
  CHECK(Finite(velocity.audio));
  CHECK(telemetry.forward_ms.size() == 1);
  CHECK(telemetry.dispatches > 0);
  CHECK(telemetry.swap_bytes == 0);

  h3::CancellationToken cancelled;
  cancelled.Cancel();
  h3::DenoiserVelocity ignored;
  CHECK(!session->Forward(0, video, audio, &cancelled, &ignored, &telemetry,
                          &error));
  CHECK(error.find("cancelled") != std::string::npos);
  error.clear();

  std::cout << "H3 denoiser smoke rows=" << session->layout().rows()
            << " active_blocks=3"
            << " refiner_ms=" << telemetry.text_refiner_ms
            << " adaln_ms=" << telemetry.adaln_precompute_ms
            << " core_load_ms=" << telemetry.core_load_ms << " peak_gib="
            << static_cast<double>(telemetry.peak_live_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " forward_ms=" << telemetry.forward_ms.front() << '\n';
  return failures == 0 ? 0 : 1;
}

int CheckOracle() {
  const char* root = std::getenv("GUFO_H3_MODEL_ROOT");
  const char* golden_value = std::getenv("GUFO_H3_DENOISER_GOLDEN");
  if (root == nullptr || golden_value == nullptr) {
    std::cout << "SKIP: GUFO_H3_MODEL_ROOT or GUFO_H3_DENOISER_GOLDEN unset\n";
    return 77;
  }
  const std::filesystem::path golden(golden_value);
  std::vector<std::pair<std::string_view, std::string_view>> hashes = {
      {"conditioning.bf16",
       "8015af1d2a551a0509bccc84372a30087f66721613f909c3f99e36ad7e269692"},
      {"refined_text.bf16",
       "be0d360ced628aa8778c95574a18f4593f938bdd0b72b292e1a8d452641e7aba"},
      {"block0_modulation.bf16",
       "38b0e3db26f24730a21304d5fa13fa049b6c5196c925cf5e87114505a1e87d39"},
      {"row_map_step0.u32",
       "816629a33e18daef4f94301aba7b54ba832c50db86fd8ae16a66f86f8d03f5d0"},
      {"video_initial.f32",
       "fce1ea483ab56dc3de354d0c056f05020ae80aba70450046b08d8ac12d9bd463"},
      {"audio_initial.f32",
       "d1c2934f2556c98e12aa0d9de550b0801037b8fea293946112d608132ca80b5f"},
      {"video_velocity_step0.f32",
       "2ae88a7d019cf64b6fc0d3a8d17e685e510304cf0e1a373a01acbf27db3d995b"},
      {"audio_velocity_step0.f32",
       "b17813b33cd76383b4614f9e025b4ecdfdad7ba60a204508d1a3ded25ada9079"},
  };
  for (const auto& [path, hash] : hashes) {
    CHECK(h3::Sha256File(golden / path) == hash);
  }
  constexpr std::size_t kTextRows = 6;
  constexpr std::size_t kVideoElements = 24 * 7 * 16 * 16;
  constexpr std::size_t kAudioElements = 32 * 2 * 37;
  constexpr std::size_t kBlockModulationElements =
      7 * 3 * h3::kDitModulationSlots * h3::kDitHiddenSize;
  auto conditioning = ReadBf16(golden / "conditioning.bf16",
                               kTextRows * h3::kH3TextConditionWidth);
  auto expected_refined =
      ReadBf16(golden / "refined_text.bf16", kTextRows * h3::kDitHiddenSize);
  auto expected_modulation =
      ReadBf16(golden / "block0_modulation.bf16", kBlockModulationElements);
  auto video_initial = ReadF32(golden / "video_initial.f32", kVideoElements);
  auto audio_initial = ReadF32(golden / "audio_initial.f32", kAudioElements);
  auto expected_video_velocity =
      ReadF32(golden / "video_velocity_step0.f32", kVideoElements);
  auto expected_audio_velocity =
      ReadF32(golden / "audio_velocity_step0.f32", kAudioElements);
  CHECK(!conditioning.empty());
  CHECK(!expected_refined.empty());
  CHECK(!expected_modulation.empty());
  CHECK(!video_initial.empty());
  CHECK(!audio_initial.empty());
  if (failures != 0) {
    return 1;
  }

  std::string error;
  auto inventory =
      h3::ModelInventory::Inspect(root, GUFO_H3_SOURCE_MANIFEST, &error);
  auto geometry = h3::ResolveGenerationGeometry(256, 256, 22, &error);
  CHECK(inventory.has_value());
  CHECK(geometry.has_value());
  if (!inventory || !geometry) {
    std::cerr << error << '\n';
    return 1;
  }
  h3::DenoiserTelemetry creation;
  auto session = h3::DenoiserSession::Create(
      *inventory,
      {.geometry = *geometry, .evaluations = 4, .text_rows = kTextRows},
      {.layer50 = conditioning, .tags = {}}, nullptr, &creation, &error);
  CHECK(session != nullptr);
  if (session == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  CHECK(session->layout().rows() == 528);
  CHECK(creation.core_gemm_plan_validations == 4);
  CheckBf16Boundary("refined_text", session->refined_text(), expected_refined,
                    0.02);
  CheckBf16Boundary("block0_modulation", session->block0_modulation(),
                    expected_modulation, 0.02);

  h3::DenoiserVelocity velocity;
  h3::DenoiserTelemetry telemetry;
  CHECK(session->Forward(0, video_initial, audio_initial, nullptr, &velocity,
                         &telemetry, &error));
  if (velocity.video.empty()) {
    std::cerr << error << '\n';
    return 1;
  }
  CheckF32Boundary("video_velocity", velocity.video, expected_video_velocity,
                   0.04, 0.05);
  CheckF32Boundary("audio_velocity", velocity.audio, expected_audio_velocity,
                   0.05, 0.05);
  CHECK(telemetry.first_non_finite.empty());
  CHECK(telemetry.forward_ms.size() == 1);
  CHECK(telemetry.swap_bytes == 0);
  std::cout << "H3 denoiser forward oracle rows=" << session->layout().rows()
            << " setup_ms=" << telemetry.setup_ms
            << " load_ms=" << telemetry.core_load_ms
            << " adaln_ms=" << telemetry.adaln_precompute_ms << " peak_gib="
            << static_cast<double>(telemetry.peak_live_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " forward_ms=" << telemetry.forward_ms.front() << '\n';
  return failures == 0 ? 0 : 1;
}

int CheckProductionShapeOracle(bool row_parallel) {
  const char* root = std::getenv("GUFO_H3_MODEL_ROOT");
  const char* golden_value = std::getenv("GUFO_H3_DENOISER_GOLDEN_512");
  if (root == nullptr || golden_value == nullptr) {
    std::cout << "SKIP: GUFO_H3_MODEL_ROOT or "
                 "GUFO_H3_DENOISER_GOLDEN_512 unset\n";
    return 77;
  }
  const std::filesystem::path golden(golden_value);
  std::vector<std::pair<std::string_view, std::string_view>> hashes = {
      {"conditioning.bf16",
       "8015af1d2a551a0509bccc84372a30087f66721613f909c3f99e36ad7e269692"},
      {"refined_text.bf16",
       "be0d360ced628aa8778c95574a18f4593f938bdd0b72b292e1a8d452641e7aba"},
      {"block0_modulation.bf16",
       "d7acea1b164b1c93ef3681d451ea49da6c51deb1adc98c2ee6882fbb0f027045"},
      {"video_initial.f32",
       "6788a83c4b5aa596cde87c901156b6ad5982da2a6ce674d7067f16699924f293"},
      {"audio_initial.f32",
       "2edc6221f9abca6c3b9a8b07cd419bf8b61649499b92706c8987633a6a41d027"},
      {"video_velocity_step0.f32",
       "528728732d4e99d19c3f47ab0a815f93025c5970ffcbaee629ff3844c44d6ff9"},
      {"audio_velocity_step0.f32",
       "895b1b8b839b37af06f8afea0577b882c8166038dccd649509a6b19febecb052"},
  };
  for (const auto& [path, hash] : hashes) {
    CHECK(h3::Sha256File(golden / path) == hash);
  }
  constexpr std::size_t kTextRows = 6;
  constexpr std::size_t kVideoElements = 24 * 7 * 32 * 32;
  constexpr std::size_t kAudioElements = 32 * 2 * 37;
  constexpr std::size_t kBlockModulationElements =
      3 * 3 * h3::kDitModulationSlots * h3::kDitHiddenSize;
  auto conditioning = ReadBf16(golden / "conditioning.bf16",
                               kTextRows * h3::kH3TextConditionWidth);
  auto expected_refined =
      ReadBf16(golden / "refined_text.bf16", kTextRows * h3::kDitHiddenSize);
  auto expected_modulation =
      ReadBf16(golden / "block0_modulation.bf16", kBlockModulationElements);
  auto video_initial = ReadF32(golden / "video_initial.f32", kVideoElements);
  auto audio_initial = ReadF32(golden / "audio_initial.f32", kAudioElements);
  auto expected_video_velocity =
      ReadF32(golden / "video_velocity_step0.f32", kVideoElements);
  auto expected_audio_velocity =
      ReadF32(golden / "audio_velocity_step0.f32", kAudioElements);
  CHECK(!conditioning.empty());
  CHECK(!expected_refined.empty());
  CHECK(!expected_modulation.empty());
  CHECK(!video_initial.empty());
  CHECK(!audio_initial.empty());
  if (failures != 0) {
    return 1;
  }

  std::string error;
  auto inventory =
      h3::ModelInventory::Inspect(root, GUFO_H3_SOURCE_MANIFEST, &error);
  auto geometry = h3::ResolveGenerationGeometry(512, 512, 22, &error);
  CHECK(inventory.has_value());
  CHECK(geometry.has_value());
  if (!inventory || !geometry) {
    std::cerr << error << '\n';
    return 1;
  }
  h3::DenoiserTelemetry creation;
  auto session = h3::DenoiserSession::Create(
      *inventory,
      {.geometry = *geometry,
       .evaluations = 2,
       .text_rows = kTextRows,
       .row_parallel_attention = row_parallel},
      {.layer50 = conditioning, .tags = {}}, nullptr, &creation, &error);
  CHECK(session != nullptr);
  if (session == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  CHECK(session->layout().rows() == 1872);
  CheckBf16Boundary("production_refined_text", session->refined_text(),
                    expected_refined, 0.02);
  CheckBf16Boundary("production_block0_modulation",
                    session->block0_modulation(), expected_modulation, 0.02);

  h3::DenoiserVelocity velocity;
  h3::DenoiserTelemetry telemetry;
  CHECK(session->Forward(0, video_initial, audio_initial, nullptr, &velocity,
                         &telemetry, &error));
  if (velocity.video.empty()) {
    std::cerr << error << '\n';
    return 1;
  }
  CheckF32Boundary("production_video_velocity", velocity.video,
                   expected_video_velocity, 0.04, 0.05);
  CheckF32Boundary("production_audio_velocity", velocity.audio,
                   expected_audio_velocity, 0.05, 0.05);
  CHECK(telemetry.first_non_finite.empty());
  CHECK(telemetry.forward_ms.size() == 1);
  CHECK(telemetry.swap_bytes == 0);
  std::cout << "H3 production-shape denoiser forward oracle rows="
            << session->layout().rows() << " load_ms=" << telemetry.core_load_ms
            << " adaln_ms=" << telemetry.adaln_precompute_ms << " peak_gib="
            << static_cast<double>(telemetry.peak_live_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " attention="
            << (telemetry.row_parallel_attention ? "row_parallel" : "scalar")
            << " forward_ms=" << telemetry.forward_ms.front() << '\n';
  return failures == 0 ? 0 : 1;
}

int CheckReuseSmoke() {
  const char* root = std::getenv("GUFO_H3_MODEL_ROOT");
  const char* conditioning_path = std::getenv("GUFO_H3_LAYER50_CONDITIONING");
  if (root == nullptr || conditioning_path == nullptr) {
    std::cout << "SKIP: GUFO_H3_MODEL_ROOT or "
                 "GUFO_H3_LAYER50_CONDITIONING unset\n";
    return 77;
  }
  constexpr std::size_t kTextRows = 6;
  auto conditioning =
      ReadBf16(conditioning_path, kTextRows * h3::kH3TextConditionWidth);
  std::string error;
  auto inventory =
      h3::ModelInventory::Inspect(root, GUFO_H3_SOURCE_MANIFEST, &error);
  auto geometry = h3::ResolveGenerationGeometry(32, 32, 5, &error);
  CHECK(inventory.has_value());
  CHECK(geometry.has_value());
  CHECK(!conditioning.empty());
  if (!inventory || !geometry || conditioning.empty()) {
    std::cerr << error << '\n';
    return 1;
  }

  h3::DenoiserTelemetry invalid_telemetry;
  auto invalid_blocks =
      h3::DenoiserSession::Create(*inventory,
                                  {.geometry = *geometry,
                                   .evaluations = 5,
                                   .text_rows = kTextRows,
                                   .active_blocks = 0,
                                   .reuse_interval = 3},
                                  {.layer50 = conditioning, .tags = {}},
                                  nullptr, &invalid_telemetry, &error);
  CHECK(invalid_blocks == nullptr);
  CHECK(error.find("invalid") != std::string::npos);
  error.clear();
  auto invalid_reuse =
      h3::DenoiserSession::Create(*inventory,
                                  {.geometry = *geometry,
                                   .evaluations = 5,
                                   .text_rows = kTextRows,
                                   .active_blocks = 3,
                                   .reuse_interval = 33},
                                  {.layer50 = conditioning, .tags = {}},
                                  nullptr, &invalid_telemetry, &error);
  CHECK(invalid_reuse == nullptr);
  CHECK(error.find("invalid") != std::string::npos);
  error.clear();

  h3::DenoiserTelemetry creation;
  auto session = h3::DenoiserSession::Create(
      *inventory,
      {.geometry = *geometry,
       .evaluations = 5,
       .text_rows = kTextRows,
       .active_blocks = 3,
       .reuse_interval = 3},
      {.layer50 = conditioning, .tags = {}}, nullptr, &creation, &error);
  CHECK(session != nullptr);
  if (session == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  CHECK(creation.core_weight_bytes > 0);
  CHECK(creation.core_weight_bytes < 4ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK(creation.peak_live_bytes < 8ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK(creation.swap_bytes == 0);

  const std::size_t video_elements =
      h3::kH3VideoLatentChannels *
      static_cast<std::size_t>(geometry->temporal.video_latent_frames) *
      static_cast<std::size_t>(geometry->latent_height) *
      static_cast<std::size_t>(geometry->latent_width);
  const std::size_t audio_elements =
      h3::kH3AudioLatentChannels * h3::kH3AudioTracks *
      static_cast<std::size_t>(geometry->temporal.audio_latent_frames);
  std::vector<float> initial_video(video_elements);
  std::vector<float> initial_audio(audio_elements);
  for (std::size_t index = 0; index < initial_video.size(); ++index) {
    initial_video[index] = std::sin(static_cast<float>(index) * 0.031F) * 0.5F;
  }
  for (std::size_t index = 0; index < initial_audio.size(); ++index) {
    initial_audio[index] = std::cos(static_cast<float>(index) * 0.019F) * 0.5F;
  }
  std::vector<float> first_video = initial_video;
  std::vector<float> first_audio = initial_audio;
  h3::DenoiserTelemetry first;
  CHECK(session->Denoise(first_video, first_audio, nullptr, &first, &error));
  CHECK(Finite(first_video));
  CHECK(Finite(first_audio));
  CHECK(first.forward_ms.size() == 3);
  CHECK(first.dispatches == 142);
  CHECK(first.swap_bytes == 0);

  std::cout << "H3 denoiser reuse smoke active_blocks=3 steps=5 "
               "reuse=3 fresh_evaluations=3 peak_gib="
            << static_cast<double>(first.peak_live_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " sampler_ms=" << first.sampler_ms << '\n';
  return failures == 0 ? 0 : 1;
}

int ProfileFullResolutionForward() {
  const char* root = std::getenv("GUFO_H3_MODEL_ROOT");
  const char* conditioning_path = std::getenv("GUFO_H3_LAYER50_CONDITIONING");
  if (root == nullptr || conditioning_path == nullptr) {
    std::cout << "SKIP: GUFO_H3_MODEL_ROOT or "
                 "GUFO_H3_LAYER50_CONDITIONING unset\n";
    return 77;
  }
  constexpr std::size_t kTextRows = 6;
  auto conditioning =
      ReadBf16(conditioning_path, kTextRows * h3::kH3TextConditionWidth);
  std::string error;
  auto inventory =
      h3::ModelInventory::Inspect(root, GUFO_H3_SOURCE_MANIFEST, &error);
  auto geometry = h3::ResolveGenerationGeometry(1344, 768, 124, &error);
  CHECK(inventory.has_value());
  CHECK(geometry.has_value());
  CHECK(!conditioning.empty());
  if (!inventory || !geometry || conditioning.empty()) {
    std::cerr << error << '\n';
    return 1;
  }

  h3::DenoiserTelemetry creation;
  auto session = h3::DenoiserSession::Create(
      *inventory,
      {.geometry = *geometry,
       .evaluations = 2,
       .text_rows = kTextRows,
       .active_blocks = static_cast<int>(h3::kDitBlocks),
       .reuse_interval = 1},
      {.layer50 = conditioning, .tags = {}}, nullptr, &creation, &error);
  CHECK(session != nullptr);
  if (session == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  CHECK(session->layout().rows() == 37716);

  const std::size_t video_elements =
      h3::kH3VideoLatentChannels *
      static_cast<std::size_t>(geometry->temporal.video_latent_frames) *
      static_cast<std::size_t>(geometry->latent_height) *
      static_cast<std::size_t>(geometry->latent_width);
  const std::size_t audio_elements =
      h3::kH3AudioLatentChannels * h3::kH3AudioTracks *
      static_cast<std::size_t>(geometry->temporal.audio_latent_frames);
  std::vector<float> video(video_elements, 0.0F);
  std::vector<float> audio(audio_elements, 0.0F);
  h3::DenoiserVelocity velocity;
  h3::DenoiserTelemetry telemetry;
  CHECK(session->Forward(0, video, audio, nullptr, &velocity, &telemetry,
                         &error));
  if (velocity.video.empty()) {
    std::cerr << error << '\n';
    return 1;
  }
  CHECK(Finite(velocity.video));
  CHECK(Finite(velocity.audio));
  CHECK(telemetry.first_non_finite.empty());
  CHECK(telemetry.forward_ms.size() == 1);
  const auto hash = [](const std::vector<float>& values) {
    return h3::Sha256({reinterpret_cast<const unsigned char*>(values.data()),
                       values.size() * sizeof(float)});
  };
  std::cout << "H3 full-resolution denoiser profile rows="
            << session->layout().rows() << " blocks=" << h3::kDitBlocks
            << " video_sha256=" << hash(velocity.video)
            << " audio_sha256=" << hash(velocity.audio)
            << " core_load_ms=" << telemetry.core_load_ms
            << " adaln_ms=" << telemetry.adaln_precompute_ms << " scratch_gib="
            << static_cast<double>(telemetry.scratch_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " peak_gib="
            << static_cast<double>(telemetry.peak_live_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " cumulative_gib="
            << static_cast<double>(telemetry.cumulative_allocation_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " forward_ms=" << telemetry.forward_ms.front() << '\n';
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 ||
      (std::string_view(argv[1]) != "--smoke" &&
       std::string_view(argv[1]) != "--oracle-forward" &&
       std::string_view(argv[1]) != "--oracle-512-forward" &&
       std::string_view(argv[1]) != "--oracle-512-forward-scalar" &&
       std::string_view(argv[1]) != "--reuse-smoke" &&
       std::string_view(argv[1]) != "--profile-fullres-forward")) {
    std::cerr << "usage: " << argv[0]
              << " --smoke|--oracle-forward|--oracle-512-forward|"
                 "--oracle-512-forward-scalar|"
                 "--reuse-smoke|--profile-fullres-forward\n";
    return 2;
  }
  if (std::string_view(argv[1]) == "--oracle-forward") {
    return CheckOracle();
  }
  if (std::string_view(argv[1]) == "--oracle-512-forward" ||
      std::string_view(argv[1]) == "--oracle-512-forward-scalar") {
    return CheckProductionShapeOracle(std::string_view(argv[1]) !=
                                      "--oracle-512-forward-scalar");
  }
  if (std::string_view(argv[1]) == "--reuse-smoke") {
    return CheckReuseSmoke();
  }
  if (std::string_view(argv[1]) == "--profile-fullres-forward") {
    return ProfileFullResolutionForward();
  }
  return CheckReal();
}
