#include "src/cli/serve/audio_tts_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <utility>

#include "src/cli/serve/json.hpp"
#include "src/models/qwen3_tts/audio.hpp"

namespace strix::server {
namespace {

constexpr std::string_view kSpeechPath = "/v1/audio/speech";
constexpr std::string_view kVoicesPath = "/v1/audio/voices";
constexpr std::size_t kMaximumInputBytes = 16U << 10U;
constexpr std::size_t kMaximumNewTokens = 8192;

bool ReadUnsigned(const json::Value& body, std::string_view name,
                  std::size_t default_value, std::size_t maximum,
                  std::size_t* output) {
  const json::Value* value = body.find(std::string(name));
  if (value == nullptr) {
    *output = default_value;
    return true;
  }
  if (!value->is_number()) {
    return false;
  }
  const double number = value->as_double();
  if (!std::isfinite(number) || number < 0.0 || std::floor(number) != number ||
      number > static_cast<double>(maximum)) {
    return false;
  }
  *output = static_cast<std::size_t>(number);
  return true;
}

HttpResponse Error(int status, std::string reason, std::string message,
                   std::string code) {
  json::Value root = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = std::move(message);
  error["type"] = "invalid_request_error";
  error["code"] = std::move(code);
  root["error"] = std::move(error);
  return {
      .status = status,
      .reason = std::move(reason),
      .body = root.dump(),
      .headers = {},
  };
}

bool HasOnlyMembers(const json::Value& value,
                    std::initializer_list<std::string_view> allowed) {
  return std::ranges::all_of(value.members(), [&](const auto& member) {
    return std::ranges::find(allowed, member.first) != allowed.end();
  });
}

void AppendU16(std::string* output, std::uint16_t value) {
  output->push_back(static_cast<char>(value & 0xFFU));
  output->push_back(static_cast<char>((value >> 8U) & 0xFFU));
}

void AppendU32(std::string* output, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xFFU));
  }
}

std::string EncodeWav(std::span<const float> samples,
                      std::uint32_t sample_rate) {
  constexpr std::uint16_t channels = 1;
  constexpr std::uint16_t bits_per_sample = 16;
  constexpr std::uint16_t block_align =
      channels * (bits_per_sample / static_cast<std::uint16_t>(8));
  const std::size_t maximum_samples =
      std::numeric_limits<std::uint32_t>::max() / block_align;
  if (sample_rate == 0 || samples.size() > maximum_samples) {
    return {};
  }
  const auto data_bytes =
      static_cast<std::uint32_t>(samples.size() * block_align);
  std::string output;
  output.reserve(44U + data_bytes);
  output.append("RIFF", 4);
  AppendU32(&output, 36U + data_bytes);
  output.append("WAVEfmt ", 8);
  AppendU32(&output, 16);
  AppendU16(&output, 1);
  AppendU16(&output, channels);
  AppendU32(&output, sample_rate);
  AppendU32(&output, sample_rate * block_align);
  AppendU16(&output, block_align);
  AppendU16(&output, bits_per_sample);
  output.append("data", 4);
  AppendU32(&output, data_bytes);
  for (const float sample : samples) {
    const float bounded = std::clamp(sample, -1.0F, 1.0F);
    const auto pcm =
        static_cast<std::int16_t>(bounded * static_cast<float>(32767));
    AppendU16(&output, static_cast<std::uint16_t>(pcm));
  }
  return output;
}

HttpResponse Voices(TtsService& service) {
  json::Value root = json::Value::object();
  root["object"] = "list";
  root["schema"] = std::string(kAudioTtsApiSchema);
  json::Value data = json::Value::array();
  for (const std::string& voice : service.voices()) {
    json::Value item = json::Value::object();
    item["id"] = voice;
    item["object"] = "voice";
    item["model"] = service.model_id();
    data.push_back(std::move(item));
  }
  root["data"] = std::move(data);
  return {
      .status = 200,
      .reason = "OK",
      .body = root.dump(),
      .headers = {},
  };
}

HttpResponse Speech(const HttpRequest& request, TtsService& service) {
  json::Value body;
  try {
    body = json::parse(request.body);
  } catch (const std::exception&) {
    return Error(400, "Bad Request", "request body must be valid JSON",
                 "parse_error");
  }
  if (!body.is_object() ||
      !HasOnlyMembers(
          body, {"model", "input", "voice", "response_format", "speed",
                 "language", "instruct", "seed", "max_new_tokens", "greedy",
                 "reference_audio", "reference_text", "voice_clone_mode"})) {
    return Error(400, "Bad Request",
                 "request contains unsupported audio speech fields",
                 "unsupported_field");
  }
  const json::Value* model = body.find("model");
  const json::Value* input = body.find("input");
  const json::Value* voice = body.find("voice");
  if (model == nullptr || !model->is_string() || input == nullptr ||
      !input->is_string() || (voice != nullptr && !voice->is_string())) {
    return Error(400, "Bad Request",
                 "'model' and 'input' must be strings; 'voice' must be a "
                 "string when supplied",
                 "invalid_parameter_type");
  }
  if (model->str() != service.model_id() && model->str() != "qwen3-tts") {
    return Error(400, "Bad Request", "unsupported audio speech model",
                 "invalid_model");
  }
  if (input->str().empty()) {
    return Error(400, "Bad Request", "'input' must not be empty",
                 "missing_input");
  }
  if (input->str().size() > kMaximumInputBytes) {
    return Error(413, "Payload Too Large",
                 "audio speech input exceeds 16384 UTF-8 bytes",
                 "input_too_large");
  }
  const std::vector<std::string> voices = service.voices();
  if (service.variant() == models::qwen3_tts::ModelVariant::kCustomVoice &&
      voice == nullptr) {
    return Error(400, "Bad Request",
                 "CustomVoice requests require a 'voice' speaker",
                 "missing_voice");
  }
  const std::string selected_voice =
      voice != nullptr ? voice->str()
                       : (voices.empty() ? std::string{} : voices.front());
  if (std::ranges::find(voices, selected_voice) == voices.end()) {
    return Error(400, "Bad Request", "unsupported Qwen3-TTS voice",
                 "invalid_voice");
  }
  if (service.variant() == models::qwen3_tts::ModelVariant::kVoiceDesign &&
      body.member_str("instruct").empty()) {
    return Error(400, "Bad Request",
                 "VoiceDesign requests require a non-empty 'instruct'",
                 "missing_instruction");
  }
  models::qwen3_tts::AudioBuffer reference_audio;
  std::string reference_text;
  bool speaker_embedding_only = false;
  if (service.variant() == models::qwen3_tts::ModelVariant::kBase) {
    const json::Value* encoded_audio = body.find("reference_audio");
    const json::Value* clone_text = body.find("reference_text");
    const json::Value* clone_mode = body.find("voice_clone_mode");
    if (encoded_audio == nullptr || !encoded_audio->is_string()) {
      return Error(400, "Bad Request",
                   "Base voice cloning requires base64 WAV "
                   "'reference_audio'",
                   "missing_reference_audio");
    }
    if (clone_mode != nullptr &&
        (!clone_mode->is_string() ||
         (clone_mode->str() != "icl" &&
          clone_mode->str() != "speaker_embedding_only"))) {
      return Error(400, "Bad Request",
                   "'voice_clone_mode' must be 'icl' or "
                   "'speaker_embedding_only'",
                   "invalid_voice_clone_mode");
    }
    speaker_embedding_only =
        clone_mode != nullptr && clone_mode->str() == "speaker_embedding_only";
    if (!speaker_embedding_only &&
        (clone_text == nullptr || !clone_text->is_string() ||
         clone_text->str().empty())) {
      return Error(400, "Bad Request",
                   "Base ICL voice cloning requires non-empty "
                   "'reference_text'",
                   "missing_reference_text");
    }
    if (clone_text != nullptr && !clone_text->is_string()) {
      return Error(400, "Bad Request", "'reference_text' must be a string",
                   "invalid_parameter_type");
    }
    std::string audio_error;
    if (!models::qwen3_tts::DecodeBase64Wav(encoded_audio->str(),
                                            &reference_audio, &audio_error)) {
      return Error(400, "Bad Request", std::move(audio_error),
                   "invalid_reference_audio");
    }
    reference_text = clone_text == nullptr ? std::string{} : clone_text->str();
  } else if (body.find("reference_audio") != nullptr ||
             body.find("reference_text") != nullptr ||
             body.find("voice_clone_mode") != nullptr) {
    return Error(400, "Bad Request",
                 "reference audio fields are valid only for the Base model",
                 "unsupported_field");
  }
  if (const json::Value* format = body.find("response_format");
      format != nullptr && (!format->is_string() || format->str() != "wav")) {
    return Error(400, "Bad Request",
                 "Qwen3-TTS currently supports response_format 'wav'",
                 "invalid_response_format");
  }
  if (const json::Value* speed = body.find("speed");
      speed != nullptr && (!speed->is_number() || speed->as_double() != 1.0)) {
    return Error(400, "Bad Request",
                 "Qwen3-TTS currently supports speed 1.0 only",
                 "invalid_speed");
  }
  for (const std::string_view name : {"language", "instruct"}) {
    const json::Value* value = body.find(std::string(name));
    if (value != nullptr && !value->is_string()) {
      return Error(400, "Bad Request",
                   "'" + std::string(name) + "' must be a string",
                   "invalid_parameter_type");
    }
  }
  std::size_t max_new_tokens = 0;
  if (!ReadUnsigned(body, "max_new_tokens", 3000, kMaximumNewTokens,
                    &max_new_tokens) ||
      max_new_tokens == 0) {
    return Error(400, "Bad Request",
                 "'max_new_tokens' must be between 1 and 8192",
                 "invalid_max_new_tokens");
  }
  std::size_t seed = 0;
  if (!ReadUnsigned(body, "seed", 42, std::numeric_limits<std::uint32_t>::max(),
                    &seed)) {
    return Error(400, "Bad Request", "'seed' must be a uint32", "invalid_seed");
  }
  bool greedy = false;
  if (const json::Value* value = body.find("greedy")) {
    if (!value->is_bool()) {
      return Error(400, "Bad Request", "'greedy' must be a boolean",
                   "invalid_greedy");
    }
    greedy = value->as_bool();
  }

  const models::qwen3_tts::SynthesisRequest synthesis{
      .text = input->str(),
      .speaker = selected_voice,
      .language = body.member_str("language", "english"),
      .instruct = body.member_str("instruct"),
      .reference_audio = std::move(reference_audio),
      .reference_text = std::move(reference_text),
      .speaker_embedding_only = speaker_embedding_only,
      .max_new_tokens = max_new_tokens,
      .seed = static_cast<std::uint32_t>(seed),
      .greedy = greedy,
  };
  models::qwen3_tts::SynthesisResult result;
  std::string error;
  if (!service.Synthesize(synthesis, request.is_cancelled, &result, &error)) {
    if (request.is_cancelled && request.is_cancelled()) {
      return Error(499, "Client Closed Request", "audio generation cancelled",
                   "cancelled");
    }
    return Error(500, "Internal Server Error", std::move(error),
                 "generation_failed");
  }
  std::string wav = EncodeWav(result.samples, result.sample_rate);
  if (wav.empty()) {
    return Error(500, "Internal Server Error",
                 "Qwen3-TTS produced invalid audio", "invalid_audio");
  }
  return {
      .status = 200,
      .reason = "OK",
      .body = std::move(wav),
      .headers =
          {
              {"Content-Type", "audio/wav"},
              {"X-Strix-Schema", std::string(kAudioTtsApiSchema)},
              {"X-Strix-TTS-Backend", service.backend_name()},
              {"X-Strix-Codec-Steps",
               std::to_string(result.codes.size() /
                              std::max<std::uint32_t>(1, result.code_groups))},
          },
  };
}

}  // namespace

bool IsAudioTtsApiPath(std::string_view path) noexcept {
  return path == kSpeechPath || path == kVoicesPath;
}

HttpResponse HandleAudioTtsApiRequest(const HttpRequest& request,
                                      TtsService& service) {
  if (request.path == kVoicesPath && request.method == "GET") {
    return Voices(service);
  }
  if (request.path == kSpeechPath && request.method == "POST") {
    return Speech(request, service);
  }
  return Error(405, "Method Not Allowed",
               "HTTP method is not supported for this audio endpoint",
               "method_not_allowed");
}

}  // namespace strix::server
