#include "src/models/minimax_h3/generation.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <gufo/ext/platform/os_info.hpp>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <utility>

#include "src/models/minimax_h3/sampling.hpp"
#include "src/models/minimax_h3/sha256.hpp"
#include "src/models/minimax_h3/tokenizer.hpp"

namespace gufo::minimax_h3 {

std::optional<int> ResolveDurationFrames(std::string_view seconds,
                                         std::string* error) {
  double duration = 0.0;
  const auto [end, status] = std::from_chars(
      seconds.data(), seconds.data() + seconds.size(), duration);
  if (status == std::errc{} && end == seconds.data() + seconds.size() &&
      std::isfinite(duration)) {
    if (duration == 1.0)
      return 22;
    if (duration >= 5.0 && duration <= 15.0) {
      const int frames = AlignFrameCount(
          static_cast<int>(std::ceil(duration * kH3FramesPerSecond)));
      if (frames <= kH3MaximumFrames)
        return frames;
    }
  }
  if (error != nullptr)
    *error =
        "MiniMax H3 seconds must be 1 (diagnostic) or 5..15 with "
        "24-fps VAE alignment fitting within 345 frames (14.375 seconds)";
  return std::nullopt;
}

namespace {

constexpr std::string_view kModelRevision =
    "42ed227ee7df40d41602854ae760620d6eb651fe";
constexpr std::string_view kReferenceRevision =
    "8974cc055ea9c02fcd14cc27dfda3e1027c05153";

std::uint64_t NextOutputNonce() {
  static std::atomic_uint64_t nonce{0};
  return nonce.fetch_add(1, std::memory_order_relaxed);
}

std::uint8_t QuantizeRgb(float source) {
  const float value =
      std::clamp(std::isfinite(source) ? source : 0.0F, 0.0F, 1.0F);
  const float scaled = value * 255.0F;
  const float lower = std::floor(scaled);
  const float fraction = scaled - lower;
  const bool round_up =
      fraction > 0.5F ||
      (fraction == 0.5F && static_cast<unsigned int>(lower) % 2U != 0U);
  return static_cast<std::uint8_t>(lower + (round_up ? 1.0F : 0.0F));
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool IsCancelled(const CancellationToken* cancellation) {
  return cancellation != nullptr && cancellation->IsCancelled();
}

#if defined(ENGINE_ENABLE_HIP)
using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}
#endif

std::string EscapeJson(std::string_view value) {
  std::ostringstream output;
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (byte < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(byte) << std::dec;
        } else {
          output << static_cast<char>(byte);
        }
    }
  }
  return output.str();
}

bool WriteAtomic(const std::filesystem::path& path, std::string_view contents,
                 std::string* error) {
  if (path.empty()) {
    SetError(error, "MiniMax H3 output path is empty");
    return false;
  }
  std::error_code filesystem_error;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), filesystem_error);
  }
  if (filesystem_error) {
    SetError(error, "cannot create MiniMax H3 output directory: " +
                        filesystem_error.message());
    return false;
  }
  const std::filesystem::path partial = path.string() + ".gufo-partial-" +
                                        std::to_string(platform::ProcessId()) +
                                        "-" + std::to_string(NextOutputNonce());
  {
    std::ofstream output(partial, std::ios::binary | std::ios::trunc);
    output.write(contents.data(),
                 static_cast<std::streamsize>(contents.size()));
    if (!output) {
      std::filesystem::remove(partial, filesystem_error);
      SetError(error, "cannot write MiniMax H3 output file");
      return false;
    }
  }
  std::filesystem::rename(partial, path, filesystem_error);
  if (filesystem_error) {
    std::filesystem::remove(partial, filesystem_error);
    SetError(error, "cannot publish MiniMax H3 output file: " +
                        filesystem_error.message());
    return false;
  }
  return true;
}

bool WriteFramesAtomic(const std::filesystem::path& directory,
                       const VideoFrames& frames,
                       const CancellationToken* cancellation,
                       std::string* error) {
  if (directory.empty() || frames.width < 1 || frames.height < 1 ||
      frames.frame_indices.empty()) {
    SetError(error, "invalid MiniMax H3 frame output");
    return false;
  }
  const std::size_t frame_elements =
      static_cast<std::size_t>(frames.width) * frames.height * 3U;
  if (frames.rgb.size() != frame_elements * frames.frame_indices.size()) {
    SetError(error, "MiniMax H3 frame output geometry differs");
    return false;
  }
  std::error_code filesystem_error;
  const bool exists = std::filesystem::exists(directory, filesystem_error);
  if (filesystem_error) {
    SetError(error, "cannot inspect MiniMax H3 frame directory: " +
                        filesystem_error.message());
    return false;
  }
  if (exists) {
    SetError(error,
             "MiniMax H3 frame directory already exists; refusing to replace "
             "it");
    return false;
  }
  const std::filesystem::path parent = directory.parent_path().empty()
                                           ? std::filesystem::path(".")
                                           : directory.parent_path();
  std::filesystem::create_directories(parent, filesystem_error);
  if (filesystem_error) {
    SetError(error, "cannot create MiniMax H3 frame parent directory: " +
                        filesystem_error.message());
    return false;
  }
  const std::filesystem::path partial =
      parent / (directory.filename().string() + ".gufo-partial-" +
                std::to_string(platform::ProcessId()) + "-" +
                std::to_string(NextOutputNonce()));
  std::filesystem::remove_all(partial, filesystem_error);
  filesystem_error.clear();
  std::filesystem::create_directories(partial, filesystem_error);
  if (filesystem_error) {
    SetError(error, "cannot create MiniMax H3 partial frame directory: " +
                        filesystem_error.message());
    return false;
  }
  for (std::size_t frame = 0; frame < frames.frame_indices.size(); ++frame) {
    if (IsCancelled(cancellation)) {
      std::filesystem::remove_all(partial, filesystem_error);
      SetError(error, "MiniMax H3 frame publication cancelled");
      return false;
    }
    std::string ppm = "P6\n" + std::to_string(frames.width) + " " +
                      std::to_string(frames.height) + "\n255\n";
    ppm.resize(ppm.size() + frame_elements);
    unsigned char* pixels = reinterpret_cast<unsigned char*>(
        ppm.data() + ppm.size() - frame_elements);
    const float* source = frames.rgb.data() + frame * frame_elements;
    for (std::size_t index = 0; index < frame_elements; ++index) {
      pixels[index] = QuantizeRgb(source[index]);
    }
    std::ostringstream name;
    name << "frame-" << std::setw(4) << std::setfill('0')
         << frames.frame_indices[frame] << ".ppm";
    if (!WriteAtomic(partial / name.str(), ppm, error)) {
      std::filesystem::remove_all(partial, filesystem_error);
      return false;
    }
  }
  if (IsCancelled(cancellation)) {
    std::filesystem::remove_all(partial, filesystem_error);
    SetError(error, "MiniMax H3 frame publication cancelled");
    return false;
  }
  std::filesystem::rename(partial, directory, filesystem_error);
  if (filesystem_error) {
    std::filesystem::remove_all(partial, filesystem_error);
    SetError(error, "cannot publish MiniMax H3 frame directory: " +
                        filesystem_error.message());
    return false;
  }
  return true;
}

struct LatentManifestContext {
  const GenerationParameters* parameters{nullptr};
  std::uint64_t seed{0};
  std::string prompt_sha256;
  std::string conditioning_sha256;
};

bool WriteLatentsAtomic(const std::filesystem::path& directory,
                        const GenerationGeometry& geometry,
                        std::span<const float> video,
                        std::span<const float> audio,
                        const LatentManifestContext* context,
                        const CancellationToken* cancellation,
                        std::string* error) {
  const std::size_t expected_video =
      kH3VideoLatentChannels *
      static_cast<std::size_t>(geometry.temporal.video_latent_frames) *
      static_cast<std::size_t>(geometry.latent_height) *
      static_cast<std::size_t>(geometry.latent_width);
  const std::size_t expected_audio =
      kH3AudioLatentChannels * kH3AudioTracks *
      static_cast<std::size_t>(geometry.temporal.audio_latent_frames);
  if (directory.empty() || video.size() != expected_video ||
      audio.size() != expected_audio) {
    SetError(error, "invalid MiniMax H3 latent output");
    return false;
  }
  std::error_code filesystem_error;
  const bool exists = std::filesystem::exists(directory, filesystem_error);
  if (filesystem_error || exists) {
    SetError(error, filesystem_error
                        ? "cannot inspect MiniMax H3 latent directory: " +
                              filesystem_error.message()
                        : "MiniMax H3 latent directory already exists; "
                          "refusing to replace it");
    return false;
  }
  const std::filesystem::path parent = directory.parent_path().empty()
                                           ? std::filesystem::path(".")
                                           : directory.parent_path();
  std::filesystem::create_directories(parent, filesystem_error);
  if (filesystem_error) {
    SetError(error, "cannot create MiniMax H3 latent parent directory: " +
                        filesystem_error.message());
    return false;
  }
  const std::filesystem::path partial =
      parent / (directory.filename().string() + ".gufo-partial-" +
                std::to_string(platform::ProcessId()) + "-" +
                std::to_string(NextOutputNonce()));
  std::filesystem::remove_all(partial, filesystem_error);
  filesystem_error.clear();
  std::filesystem::create_directories(partial, filesystem_error);
  if (filesystem_error) {
    SetError(error, "cannot create MiniMax H3 partial latent directory: " +
                        filesystem_error.message());
    return false;
  }
  const auto bytes = [](std::span<const float> values) {
    return std::span<const unsigned char>(
        reinterpret_cast<const unsigned char*>(values.data()),
        values.size_bytes());
  };
  const auto write = [&](std::string_view name, std::span<const float> values) {
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 latent publication cancelled");
      return false;
    }
    const auto payload = bytes(values);
    return WriteAtomic(
        partial / name,
        std::string_view(reinterpret_cast<const char*>(payload.data()),
                         payload.size()),
        error);
  };
  if (!write("video_final.f32", video) || !write("audio_final.f32", audio)) {
    std::filesystem::remove_all(partial, filesystem_error);
    return false;
  }
  std::ostringstream manifest;
  manifest << "{\n"
           << "  \"schema\": \"gufo.minimax-h3-final-latents.v1\",\n"
           << "  \"video\": {\"dtype\": \"F32\", \"shape\": ["
           << kH3VideoLatentChannels << ", "
           << geometry.temporal.video_latent_frames << ", "
           << geometry.latent_height << ", " << geometry.latent_width
           << "], \"bytes\": " << video.size_bytes() << ", \"sha256\": \""
           << Sha256(bytes(video)) << "\"},\n"
           << "  \"audio\": {\"dtype\": \"F32\", \"shape\": ["
           << kH3AudioLatentChannels << ", " << kH3AudioTracks << ", "
           << geometry.temporal.audio_latent_frames
           << "], \"bytes\": " << audio.size_bytes() << ", \"sha256\": \""
           << Sha256(bytes(audio)) << "\"}";
  if (context != nullptr && context->parameters != nullptr) {
    const GenerationParameters& parameters = *context->parameters;
    manifest << ",\n"
             << "  \"model_revision\": \"" << kModelRevision << "\",\n"
             << "  \"reference_revision\": \"" << kReferenceRevision << "\",\n"
             << "  \"seed\": " << context->seed << ",\n"
             << "  \"noise_mode\": \"seeded\",\n"
             << "  \"prompt_sha256\": \"" << EscapeJson(context->prompt_sha256)
             << "\",\n"
             << "  \"conditioning_sha256\": \""
             << EscapeJson(context->conditioning_sha256) << "\",\n"
             << "  \"geometry\": {\"width\": " << geometry.width
             << ", \"height\": " << geometry.height
             << ", \"frames\": " << geometry.temporal.frames << "},\n"
             << "  \"steps\": " << parameters.evaluations << ",\n"
             << "  \"blocks\": " << parameters.active_blocks << ",\n"
             << "  \"reuse_interval\": " << parameters.reuse_interval << ",\n"
             << "  \"attention_kernel\": \""
             << (parameters.row_parallel_attention ? "row_parallel" : "scalar")
             << "\"";
  }
  manifest << "\n}\n";
  if (IsCancelled(cancellation) ||
      !WriteAtomic(partial / "manifest.json", manifest.str(), error)) {
    std::filesystem::remove_all(partial, filesystem_error);
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 latent publication cancelled");
    }
    return false;
  }
  std::filesystem::rename(partial, directory, filesystem_error);
  if (filesystem_error) {
    std::filesystem::remove_all(partial, filesystem_error);
    SetError(error, "cannot publish MiniMax H3 latent directory: " +
                        filesystem_error.message());
    return false;
  }
  return true;
}

#if defined(ENGINE_ENABLE_HIP)
std::vector<std::uint8_t> ToRgb24(const VideoFrames& frames) {
  std::vector<std::uint8_t> result(frames.rgb.size());
  std::transform(frames.rgb.begin(), frames.rgb.end(), result.begin(),
                 [](float source) { return QuantizeRgb(source); });
  return result;
}

struct ProgressBridge {
  GenerationProgress callback{nullptr};
  void* opaque{nullptr};
  std::string_view phase;
};

void ForwardDenoiserProgress(int completed, int total, void* opaque) {
  auto* bridge = static_cast<ProgressBridge*>(opaque);
  if (bridge != nullptr && bridge->callback != nullptr) {
    bridge->callback(bridge->phase, completed, total, bridge->opaque);
  }
}

void ForwardVideoProgress(int completed, int total, void* opaque) {
  ForwardDenoiserProgress(completed, total, opaque);
}

void ForwardAudioProgress(int completed, int total, void* opaque) {
  ForwardDenoiserProgress(completed, total, opaque);
}

void Report(GenerationProgress progress, void* opaque, std::string_view phase,
            int completed, int total) {
  if (progress != nullptr) {
    progress(phase, completed, total, opaque);
  }
}
#endif

}  // namespace

std::optional<GenerationParameters> ResolveGenerationPreset(
    std::string_view name, std::string* error) {
  GenerationParameters parameters;
  if (name == "exact" || name == "exact-512") {
    parameters = {.preset = "exact-512",
                  .internal_width = 512,
                  .internal_height = 512,
                  .output_width = 512,
                  .output_height = 512,
                  .frames = 22,
                  .evaluations = 49,
                  .active_blocks = 50,
                  .reuse_interval = 1,
                  .selected_frames = {}};
  } else if (name == "exact-1344x768" || name == "fullres") {
    parameters = {.preset = "exact-1344x768",
                  .internal_width = 1344,
                  .internal_height = 768,
                  .output_width = 1344,
                  .output_height = 768,
                  .frames = 124,
                  .evaluations = 49,
                  .active_blocks = 50,
                  .reuse_interval = 1,
                  .selected_frames = {}};
  } else if (name == "fast" || name == "fast-384") {
    parameters = {.preset = "fast-384",
                  .internal_width = 384,
                  .internal_height = 384,
                  .output_width = 512,
                  .output_height = 512,
                  .frames = 22,
                  .evaluations = 19,
                  .active_blocks = 45,
                  .reuse_interval = 2,
                  .selected_frames = {}};
  } else if (name == "aggressive" || name == "aggressive-320") {
    parameters = {.preset = "aggressive-320",
                  .internal_width = 320,
                  .internal_height = 320,
                  .output_width = 512,
                  .output_height = 512,
                  .frames = 22,
                  .evaluations = 19,
                  .active_blocks = 40,
                  .reuse_interval = 3,
                  .selected_frames = {}};
  } else if (name == "dev" || name == "development-256") {
    parameters = {.preset = "development-256",
                  .internal_width = 256,
                  .internal_height = 256,
                  .output_width = 256,
                  .output_height = 256,
                  .frames = 22,
                  .evaluations = 4,
                  .active_blocks = 50,
                  .reuse_interval = 1,
                  .decode_audio = false,
                  .mux = false,
                  .selected_frames = {0, 11, 21}};
  } else {
    SetError(error,
             "unknown MiniMax H3 preset; expected exact, exact-1344x768, "
             "fast, aggressive, or dev");
    return std::nullopt;
  }
  return parameters;
}

bool ValidateGenerationParameters(const GenerationParameters& parameters,
                                  std::string* error) {
  if (parameters.preset.empty() || parameters.output_width < 2 ||
      parameters.output_height < 2 || (parameters.output_width & 1) != 0 ||
      (parameters.output_height & 1) != 0 || parameters.evaluations < 2 ||
      parameters.evaluations > kH3MaximumEvaluations ||
      parameters.active_blocks < 3 || parameters.active_blocks > 50 ||
      parameters.reuse_interval < 1 || parameters.reuse_interval > 32) {
    SetError(error, "invalid MiniMax H3 generation parameters");
    return false;
  }
  std::string geometry_error;
  const auto geometry = ResolveGenerationGeometry(
      parameters.internal_width, parameters.internal_height, parameters.frames,
      &geometry_error);
  if (!geometry.has_value() || geometry->temporal.frames != parameters.frames) {
    SetError(error, geometry_error.empty()
                        ? "MiniMax H3 generation requires a legal frame count"
                        : geometry_error);
    return false;
  }
  if (!ResolveVideoVaePlan(*geometry, &geometry_error).has_value()) {
    SetError(error, geometry_error.empty()
                        ? "MiniMax H3 generation requires decodable VisualVAE "
                          "geometry"
                        : geometry_error);
    return false;
  }
  int previous = -1;
  for (const int frame : parameters.selected_frames) {
    if (frame <= previous || frame < 0 || frame >= parameters.frames) {
      SetError(error,
               "MiniMax H3 selected frames must be increasing and in range");
      return false;
    }
    previous = frame;
  }
  if (parameters.mux &&
      (!parameters.decode_audio || !parameters.selected_frames.empty())) {
    SetError(error,
             "MiniMax H3 MP4 output requires audio and every video frame");
    return false;
  }
  return true;
}

std::string GenerationParametersJson(const GenerationParameters& parameters,
                                     std::uint64_t seed,
                                     std::string_view prompt_sha256) {
  std::ostringstream output;
  output << "{\n"
         << "  \"schema\": \"" << kH3GenerationSchema << "\",\n"
         << "  \"backend\": \"rocm-hip-gfx1151\",\n"
         << "  \"precision\": \"bf16-f32\",\n"
         << "  \"model_repository\": \"MiniMaxAI/MiniMax-H3\",\n"
         << "  \"model_revision\": \"" << kModelRevision << "\",\n"
         << "  \"reference_repository\": \"antirez/h3.c\",\n"
         << "  \"reference_revision\": \"" << kReferenceRevision << "\",\n"
         << "  \"preset\": \"" << EscapeJson(parameters.preset) << "\",\n"
         << "  \"prompt_sha256\": \"" << EscapeJson(prompt_sha256) << "\",\n"
         << "  \"seed\": " << seed << ",\n"
         << "  \"internal_width\": " << parameters.internal_width << ",\n"
         << "  \"internal_height\": " << parameters.internal_height << ",\n"
         << "  \"output_width\": " << parameters.output_width << ",\n"
         << "  \"output_height\": " << parameters.output_height << ",\n"
         << "  \"fps\": " << kH3FramesPerSecond << ",\n"
         << "  \"frames\": " << parameters.frames << ",\n"
         << "  \"evaluations\": " << parameters.evaluations << ",\n"
         << "  \"active_blocks\": " << parameters.active_blocks << ",\n"
         << "  \"reuse_interval\": " << parameters.reuse_interval << ",\n"
         << "  \"attention_kernel\": \""
         << (parameters.row_parallel_attention ? "row_parallel" : "scalar")
         << "\",\n"
         << "  \"token_reduction\": false,\n"
         << "  \"decode_audio\": "
         << (parameters.decode_audio ? "true" : "false") << ",\n"
         << "  \"mux\": " << (parameters.mux ? "true" : "false") << ",\n"
         << "  \"selected_frames\": [";
  for (std::size_t index = 0; index < parameters.selected_frames.size();
       ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << parameters.selected_frames[index];
  }
  output << "]\n}\n";
  return output.str();
}

bool WriteDiagnosticFrames(const std::filesystem::path& directory,
                           const VideoFrames& frames,
                           const CancellationToken* cancellation,
                           std::string* error) {
  return WriteFramesAtomic(directory, frames, cancellation, error);
}

bool WriteDiagnosticLatents(const std::filesystem::path& directory,
                            const GenerationGeometry& geometry,
                            std::span<const float> video,
                            std::span<const float> audio,
                            const CancellationToken* cancellation,
                            std::string* error) {
  return WriteLatentsAtomic(directory, geometry, video, audio, nullptr,
                            cancellation, error);
}

bool GenerateTextVideo(const GenerationRequest& request,
                       const CancellationToken* cancellation,
                       GenerationProgress progress, void* progress_opaque,
                       GenerationTelemetry* telemetry, std::string* error) {
  if (telemetry != nullptr) {
    *telemetry = {};
  }
#if !defined(ENGINE_ENABLE_HIP)
  (void)request;
  (void)cancellation;
  (void)progress;
  (void)progress_opaque;
  SetError(error,
           "MiniMax H3 generation requires a ROCm HIP-enabled build; no CPU "
           "inference fallback is provided");
  return false;
#else
  if (request.model_root.empty() || request.source_manifest.empty() ||
      request.prompt.empty() ||
      !ValidateGenerationParameters(request.parameters, error) ||
      (request.parameters.mux && request.output_path.empty()) ||
      (!request.parameters.mux && request.frames_directory.empty()) ||
      IsCancelled(cancellation)) {
    if (error != nullptr && error->empty()) {
      SetError(error, "invalid MiniMax H3 text generation request");
    }
    return false;
  }
  GenerationTelemetry local;
  const auto total_begin = Clock::now();
  Report(progress, progress_opaque, "inventory", 0, 1);
  auto begin = Clock::now();
  auto inventory = ModelInventory::Inspect(request.model_root,
                                           request.source_manifest, error);
  local.inventory_ms = Milliseconds(begin, Clock::now());
  if (!inventory.has_value()) {
    return false;
  }
  local.inventory = inventory->telemetry();
  Report(progress, progress_opaque, "inventory", 1, 1);

  begin = Clock::now();
  Tokenizer tokenizer;
  std::vector<std::uint32_t> tokens;
  if (!Tokenizer::Load(request.model_root / "FL2VA/tokenizer/tokenizer.json",
                       &tokenizer, error) ||
      !tokenizer.EncodePrompt(request.prompt, &tokens, error) ||
      tokens.empty()) {
    return false;
  }
  local.tokenizer_ms = Milliseconds(begin, Clock::now());
  local.prompt_tokens = tokens.size();

  Report(progress, progress_opaque, "prompt", 0, 1);
  begin = Clock::now();
  PromptEmbedding embedding;
  if (!EncodePromptLayer50(*inventory, tokens, {}, cancellation, &embedding,
                           &local.prompt_encoder, error)) {
    return false;
  }
  local.prompt_ms = Milliseconds(begin, Clock::now());
  Report(progress, progress_opaque, "prompt", 1, 1);

  const auto geometry = ResolveGenerationGeometry(
      request.parameters.internal_width, request.parameters.internal_height,
      request.parameters.frames, error);
  if (!geometry.has_value()) {
    return false;
  }
  const std::size_t video_elements =
      kH3VideoLatentChannels *
      static_cast<std::size_t>(geometry->temporal.video_latent_frames) *
      static_cast<std::size_t>(geometry->latent_height) *
      static_cast<std::size_t>(geometry->latent_width);
  auto noise =
      BuildInitialNoise(request.seed, video_elements, kH3AudioLatentChannels,
                        geometry->temporal.audio_latent_frames, error);
  if (!noise.has_value()) {
    return false;
  }
  std::vector<std::uint8_t> tags(embedding.tokens, 1);
  auto denoiser = DenoiserSession::Create(
      *inventory,
      {.geometry = *geometry,
       .evaluations = request.parameters.evaluations,
       .text_rows = embedding.tokens,
       .active_blocks = request.parameters.active_blocks,
       .reuse_interval = request.parameters.reuse_interval,
       .row_parallel_attention = request.parameters.row_parallel_attention},
      {.layer50 = embedding.values, .tags = tags}, cancellation,
      &local.denoiser, error);
  if (denoiser == nullptr) {
    return false;
  }
  ProgressBridge denoiser_progress{progress, progress_opaque, "denoiser"};
  begin = Clock::now();
  if (!denoiser->DenoiseWithProgress(
          noise->video, noise->audio, cancellation, ForwardDenoiserProgress,
          &denoiser_progress, &local.denoiser, error)) {
    return false;
  }
  local.denoiser_ms = Milliseconds(begin, Clock::now());
  denoiser.reset();
  if (!request.latents_directory.empty()) {
    Report(progress, progress_opaque, "latents", 0, 1);
    const std::span<const unsigned char> prompt_bytes(
        reinterpret_cast<const unsigned char*>(request.prompt.data()),
        request.prompt.size());
    const std::span<const unsigned char> conditioning_bytes(
        reinterpret_cast<const unsigned char*>(embedding.values.data()),
        embedding.values.size() * sizeof(std::uint16_t));
    const LatentManifestContext context{
        .parameters = &request.parameters,
        .seed = request.seed,
        .prompt_sha256 = Sha256(prompt_bytes),
        .conditioning_sha256 = Sha256(conditioning_bytes),
    };
    if (!WriteLatentsAtomic(request.latents_directory, *geometry, noise->video,
                            noise->audio, &context, cancellation, error)) {
      return false;
    }
    Report(progress, progress_opaque, "latents", 1, 1);
  }

  begin = Clock::now();
  auto video_decoder =
      VideoVaeDecoder::Create(*inventory, {.geometry = *geometry}, cancellation,
                              &local.video_vae, error);
  if (video_decoder == nullptr) {
    return false;
  }
  ProgressBridge video_progress{progress, progress_opaque, "video_vae"};
  VideoFrames frames;
  const bool video_ok =
      request.parameters.selected_frames.empty()
          ? video_decoder->Decode(noise->video, cancellation,
                                  ForwardVideoProgress, &video_progress,
                                  &frames, &local.video_vae, error)
          : video_decoder->DecodeSelected(
                noise->video, request.parameters.selected_frames, cancellation,
                ForwardVideoProgress, &video_progress, &frames,
                &local.video_vae, error);
  if (!video_ok) {
    return false;
  }
  local.video_vae_ms = Milliseconds(begin, Clock::now());
  local.first_preview_ms = Milliseconds(total_begin, Clock::now());
  video_decoder.reset();

  if (!request.frames_directory.empty() &&
      !WriteDiagnosticFrames(request.frames_directory, frames, cancellation,
                             error)) {
    return false;
  }

  if (request.parameters.decode_audio) {
    begin = Clock::now();
    auto audio_decoder = AudioVaeDecoder::Create(
        *inventory, geometry->temporal.audio_latent_frames, cancellation,
        &local.audio_vae, error);
    if (audio_decoder == nullptr) {
      return false;
    }
    ProgressBridge audio_progress{progress, progress_opaque, "audio_vae"};
    AudioWaveform waveform;
    if (!audio_decoder->Decode(noise->audio, cancellation, ForwardAudioProgress,
                               &audio_progress, &waveform, &local.audio_vae,
                               error)) {
      return false;
    }
    local.audio_vae_ms = Milliseconds(begin, Clock::now());
    audio_decoder.reset();
    if (request.parameters.mux) {
      begin = Clock::now();
      const std::vector<std::uint8_t> rgb24 = ToRgb24(frames);
      if (!WriteSynchronizedMp4(
              request.output_path, rgb24,
              static_cast<int>(frames.frame_indices.size()),
              {.input_width = frames.width,
               .input_height = frames.height,
               .output_width = request.parameters.output_width,
               .output_height = request.parameters.output_height,
               .fps = kH3FramesPerSecond},
              waveform.pcm, waveform.samples, waveform.channels,
              waveform.sample_rate, cancellation, &local.media, error)) {
        return false;
      }
      local.media_ms = Milliseconds(begin, Clock::now());
    }
  }
  local.total_ms = Milliseconds(total_begin, Clock::now());
  if (telemetry != nullptr) {
    *telemetry = std::move(local);
  }
  return true;
#endif
}

std::string GenerationTelemetryJson(const GenerationTelemetry& telemetry) {
  std::uint64_t prompt_weight_bytes = 0;
  double prompt_load_ms = 0.0;
  double prompt_prefetch_wait_ms = 0.0;
  double prompt_submit_ms = 0.0;
  double prompt_gpu_ms = 0.0;
  for (const PromptEncoderLayerTelemetry& layer :
       telemetry.prompt_encoder.layers) {
    prompt_weight_bytes += layer.weight_bytes;
    prompt_load_ms += layer.load_ms;
    prompt_prefetch_wait_ms += layer.prefetch_wait_ms;
    prompt_submit_ms += layer.submit_ms;
    prompt_gpu_ms += layer.gpu_ms;
  }
  std::ostringstream output;
  output << std::fixed << std::setprecision(3) << "{\n"
         << "  \"schema\": \"gufo.minimax-h3-generation-telemetry.v1\",\n"
         << "  \"prompt_tokens\": " << telemetry.prompt_tokens << ",\n"
         << "  \"inventory_ms\": " << telemetry.inventory_ms << ",\n"
         << "  \"inventory_json_bytes\": "
         << telemetry.inventory.json_bytes_read << ",\n"
         << "  \"inventory_header_bytes\": "
         << telemetry.inventory.safetensors_header_bytes_read << ",\n"
         << "  \"inventory_payload_bytes\": "
         << telemetry.inventory.payload_bytes_read << ",\n"
         << "  \"tokenizer_ms\": " << telemetry.tokenizer_ms << ",\n"
         << "  \"prompt_ms\": " << telemetry.prompt_ms << ",\n"
         << "  \"prompt_weight_bytes\": " << prompt_weight_bytes << ",\n"
         << "  \"prompt_load_ms\": " << prompt_load_ms << ",\n"
         << "  \"prompt_prefetch_wait_ms\": " << prompt_prefetch_wait_ms
         << ",\n"
         << "  \"prompt_submit_ms\": " << prompt_submit_ms << ",\n"
         << "  \"prompt_gpu_ms\": " << prompt_gpu_ms << ",\n"
         << "  \"prompt_peak_device_bytes\": "
         << telemetry.prompt_encoder.peak_device_bytes << ",\n"
         << "  \"prompt_peak_registered_host_bytes\": "
         << telemetry.prompt_encoder.peak_registered_host_bytes << ",\n"
         << "  \"prompt_live_registered_host_bytes_after\": "
         << telemetry.prompt_encoder.live_registered_host_bytes_after << ",\n"
         << "  \"prompt_layers\": [";
  for (std::size_t index = 0; index < telemetry.prompt_encoder.layers.size();
       ++index) {
    if (index != 0) {
      output << ", ";
    }
    const PromptEncoderLayerTelemetry& layer =
        telemetry.prompt_encoder.layers[index];
    output << "{\"layer\": " << layer.layer
           << ", \"weight_bytes\": " << layer.weight_bytes
           << ", \"load_ms\": " << layer.load_ms
           << ", \"prefetch_wait_ms\": " << layer.prefetch_wait_ms
           << ", \"submit_ms\": " << layer.submit_ms
           << ", \"gpu_ms\": " << layer.gpu_ms
           << ", \"dispatches\": " << layer.dispatches << "}";
  }
  output << "],\n"
         << "  \"denoiser_ms\": " << telemetry.denoiser_ms << ",\n"
         << "  \"denoiser_setup_ms\": " << telemetry.denoiser.setup_ms << ",\n"
         << "  \"denoiser_text_refiner_ms\": "
         << telemetry.denoiser.text_refiner_ms << ",\n"
         << "  \"denoiser_adaln_precompute_ms\": "
         << telemetry.denoiser.adaln_precompute_ms << ",\n"
         << "  \"denoiser_core_load_ms\": " << telemetry.denoiser.core_load_ms
         << ",\n"
         << "  \"denoiser_sampler_ms\": " << telemetry.denoiser.sampler_ms
         << ",\n"
         << "  \"denoiser_evaluations\": "
         << telemetry.denoiser.forward_ms.size() << ",\n"
         << "  \"denoiser_forward_ms\": [";
  for (std::size_t index = 0; index < telemetry.denoiser.forward_ms.size();
       ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << telemetry.denoiser.forward_ms[index];
  }
  output << "],\n"
         << "  \"denoiser_core_weight_bytes\": "
         << telemetry.denoiser.core_weight_bytes << ",\n"
         << "  \"denoiser_persistent_bytes\": "
         << telemetry.denoiser.persistent_bytes << ",\n"
         << "  \"denoiser_scratch_bytes\": " << telemetry.denoiser.scratch_bytes
         << ",\n"
         << "  \"denoiser_cumulative_allocation_bytes\": "
         << telemetry.denoiser.cumulative_allocation_bytes << ",\n"
         << "  \"denoiser_core_gemm_plan_validations\": "
         << telemetry.denoiser.core_gemm_plan_validations << ",\n"
         << "  \"video_vae_ms\": " << telemetry.video_vae_ms << ",\n"
         << "  \"video_vae_load_ms\": " << telemetry.video_vae.load_ms << ",\n"
         << "  \"video_vae_decode_ms\": " << telemetry.video_vae.decode_ms
         << ",\n"
         << "  \"video_vae_tile_ms\": [";
  for (std::size_t index = 0; index < telemetry.video_vae.tile_ms.size();
       ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << telemetry.video_vae.tile_ms[index];
  }
  output << "],\n"
         << "  \"first_preview_ms\": " << telemetry.first_preview_ms << ",\n"
         << "  \"audio_vae_ms\": " << telemetry.audio_vae_ms << ",\n"
         << "  \"audio_vae_load_ms\": " << telemetry.audio_vae.load_ms << ",\n"
         << "  \"audio_vae_decode_ms\": " << telemetry.audio_vae.decode_ms
         << ",\n"
         << "  \"audio_vae_stage_ms\": [";
  for (std::size_t index = 0; index < telemetry.audio_vae.stage_ms.size();
       ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << telemetry.audio_vae.stage_ms[index];
  }
  output
      << "],\n"
      << "  \"media_ms\": " << telemetry.media_ms << ",\n"
      << "  \"media_maximum_audio_staging_bytes\": "
      << telemetry.media.maximum_audio_staging_bytes << ",\n"
      << "  \"media_child_status\": " << telemetry.media.child_status << ",\n"
      << "  \"media_cancelled\": "
      << (telemetry.media.cancelled ? "true" : "false") << ",\n"
      << "  \"total_ms\": " << telemetry.total_ms << ",\n"
      << "  \"peak_denoiser_bytes\": " << telemetry.denoiser.peak_live_bytes
      << ",\n"
      << "  \"peak_video_vae_bytes\": " << telemetry.video_vae.peak_live_bytes
      << ",\n"
      << "  \"peak_audio_vae_bytes\": " << telemetry.audio_vae.peak_live_bytes
      << ",\n"
      << "  \"prompt_dispatches\": "
      << telemetry.prompt_encoder.total_dispatches << ",\n"
      << "  \"denoiser_dispatches\": " << telemetry.denoiser.dispatches << ",\n"
      << "  \"denoiser_attention_kernel\": \""
      << (telemetry.denoiser.row_parallel_attention ? "row_parallel" : "scalar")
      << "\",\n"
      << "  \"video_vae_dispatches\": " << telemetry.video_vae.dispatches
      << ",\n"
      << "  \"audio_vae_dispatches\": " << telemetry.audio_vae.dispatches
      << ",\n"
      << "  \"media_video_bytes\": " << telemetry.media.video_bytes << ",\n"
      << "  \"media_audio_bytes\": " << telemetry.media.audio_bytes << ",\n"
      << "  \"minor_page_faults\": "
      << telemetry.denoiser.minor_page_faults +
             telemetry.video_vae.minor_page_faults +
             telemetry.audio_vae.minor_page_faults
      << ",\n"
      << "  \"major_page_faults\": "
      << telemetry.denoiser.major_page_faults +
             telemetry.video_vae.major_page_faults +
             telemetry.audio_vae.major_page_faults
      << ",\n"
      << "  \"swap_bytes\": "
      << std::max({telemetry.denoiser.swap_bytes,
                   telemetry.video_vae.swap_bytes,
                   telemetry.audio_vae.swap_bytes})
      << "\n}\n";
  return output.str();
}

}  // namespace gufo::minimax_h3
