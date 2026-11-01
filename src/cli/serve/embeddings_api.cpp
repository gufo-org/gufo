#include "src/cli/serve/embeddings_api.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::server {
namespace {

constexpr std::string_view kEmbeddingsPath = "/v1/embeddings";
// llama.cpp's non-OpenAI routes. Both share one handler there.
constexpr std::string_view kNativeEmbeddingsPath = "/embeddings";
constexpr std::string_view kLegacyEmbeddingPath = "/embedding";
// llama.cpp's default multimodal marker in a `prompt_string`. Its server can
// randomize the marker and report it in /props; clients that cannot read
// /props, as here, fall back to this one.
constexpr std::string_view kMediaMarker = "<__media__>";
constexpr std::size_t kMaximumInputs = 2048;
constexpr std::array<std::size_t, 4> kDimensions = {128, 256, 512, 768};

std::string_view ReasonPhrase(int status) {
  switch (status) {
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 499:
      return "Client Closed Request";
    case 503:
      return "Service Unavailable";
    default:
      return "Internal Server Error";
  }
}

HttpResponse Error(int status, std::string message, std::string code) {
  json::Value root = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = std::move(message);
  error["type"] = status < 500 ? "invalid_request_error" : "server_error";
  error["code"] = std::move(code);
  error["param"] = nullptr;
  root["error"] = std::move(error);
  return {
      .status = status,
      .reason = std::string(ReasonPhrase(status)),
      .body = root.dump(),
      .headers = {{"Content-Type", "application/json"}},
      .streaming_body = {},
      .log_details = {},
  };
}

HttpResponse BadRequest(std::string message, std::string code) {
  return Error(400, std::move(message), std::move(code));
}

std::string Base64(std::span<const std::uint8_t> bytes) {
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string output;
  output.reserve((bytes.size() + 2U) / 3U * 4U);
  for (std::size_t i = 0; i < bytes.size(); i += 3U) {
    const std::uint32_t word =
        (std::uint32_t{bytes[i]} << 16U) |
        (i + 1U < bytes.size() ? std::uint32_t{bytes[i + 1U]} << 8U : 0U) |
        (i + 2U < bytes.size() ? std::uint32_t{bytes[i + 2U]} : 0U);
    output += alphabet[(word >> 18U) & 63U];
    output += alphabet[(word >> 12U) & 63U];
    output += i + 1U < bytes.size() ? alphabet[(word >> 6U) & 63U] : '=';
    output += i + 2U < bytes.size() ? alphabet[word & 63U] : '=';
  }
  return output;
}

/// Little-endian IEEE-754 binary32, as OpenAI's base64 format specifies.
std::string EncodeBase64Floats(std::span<const float> values) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(values.size() * sizeof(float));
  for (const float value : values) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
      bytes.push_back(static_cast<std::uint8_t>((bits >> shift) & 0xFFU));
    }
  }
  return Base64(bytes);
}

/// Shortest decimal text that parses back to the same float.
void AppendFloats(std::string* out, std::span<const float> values) {
  *out += '[';
  char buffer[32];
  bool first = true;
  for (const float value : values) {
    if (!first) {
      *out += ',';
    }
    first = false;
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    out->append(buffer, result.ptr);
  }
  *out += ']';
}

/// Matryoshka truncation keeps the leading components and renormalizes them.
/// Returns false when the prefix has no usable norm.
bool TruncateAndNormalize(std::span<const float> full, std::size_t dimensions,
                          std::vector<float>* out) {
  double sum = 0.0;
  for (std::size_t i = 0; i < dimensions; ++i) {
    sum += static_cast<double>(full[i]) * static_cast<double>(full[i]);
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    return false;
  }
  const double scale = 1.0 / std::sqrt(sum);
  out->resize(dimensions);
  for (std::size_t i = 0; i < dimensions; ++i) {
    (*out)[i] = static_cast<float>(static_cast<double>(full[i]) * scale);
  }
  return true;
}

/// Whether a base64 `multimodal_data` file looks like audio, as llama.cpp's
/// mtmd helper decides (`is_audio_file`): WAV, MP3 (an ID3 tag or an MPEG
/// sync word) or FLAC. Ogg, which Gufo's audio decoder also reads, is sniffed
/// too. Anything else, including undecodable base64, goes to the image reader.
bool LooksLikeAudio(std::string_view base64) {
  constexpr std::string_view kDigits =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  if (base64.size() < 16)
    return false;
  std::array<unsigned char, 12> head{};
  for (std::size_t i = 0; i < 16; i += 4) {
    std::uint32_t bits = 0;
    for (std::size_t j = 0; j < 4; ++j) {
      const auto digit = kDigits.find(base64[i + j]);
      if (digit == std::string_view::npos)
        return false;
      bits = (bits << 6) | static_cast<std::uint32_t>(digit);
    }
    head[i / 4 * 3] = static_cast<unsigned char>(bits >> 16);
    head[i / 4 * 3 + 1] = static_cast<unsigned char>(bits >> 8);
    head[i / 4 * 3 + 2] = static_cast<unsigned char>(bits);
  }
  const auto starts = [&](std::size_t offset, std::string_view magic) {
    return std::equal(magic.begin(), magic.end(), head.begin() + offset);
  };
  return (starts(0, "RIFF") && starts(8, "WAVE")) || starts(0, "ID3") ||
         (head[0] == 0xFF && (head[1] & 0xE0) == 0xE0) || starts(0, "fLaC") ||
         starts(0, "OggS");
}

bool IsKnownField(std::string_view name, bool native) {
  return name == "input" || name == "model" || name == "encoding_format" ||
         name == "dimensions" || name == "user" || name == "truncate" ||
         name == "prompt_name" || name == "video_options" ||
         name == "mm_processor_kwargs" ||
         (native && (name == "content" || name == "embd_normalize"));
}

/// The member's value, or nullptr when it is absent or null.
const json::Value* Optional(const json::Value& root, const std::string& name) {
  const json::Value* value = root.find(name);
  return value == nullptr || value->is_null() ? nullptr : value;
}

/// `native` selects llama.cpp's `POST /embeddings` request and response form.
HttpResponse Embed(const HttpRequest& request, EmbeddingService& service,
                   bool native) {
  json::Value root;
  try {
    root = json::parse(request.body);
  } catch (const std::exception& error) {
    return BadRequest(error.what(), "parse_error");
  }
  if (!root.is_object()) {
    return BadRequest("request body must be a JSON object",
                      "invalid_request_body");
  }
  for (const auto& [name, unused] : root.members()) {
    (void)unused;
    if (!IsKnownField(name, native)) {
      return BadRequest("unsupported embeddings parameter: " + name,
                        "unsupported_field");
    }
  }

  if (const json::Value* model = Optional(root, "model"); model != nullptr) {
    if (!model->is_string()) {
      return BadRequest("'model' must be a string", "invalid_parameter_type");
    }
    if (model->str() != service.model_id()) {
      return Error(404, "requested embedding model is not loaded",
                   "model_not_found");
    }
  }

  const json::Value* input = Optional(root, "input");
  // As in llama.cpp, `input` takes precedence over the legacy `content`.
  if (input == nullptr && native)
    input = Optional(root, "content");
  if (input == nullptr) {
    return BadRequest(
        native ? "'input' or 'content' is required" : "'input' is required",
        "missing_input");
  }
  std::vector<EmbeddingInput> inputs;
  std::size_t image_count = 0;
  std::size_t audio_count = 0;
  std::size_t video_count = 0;
  const auto parse_item = [&](const json::Value& item) -> HttpResponse {
    EmbeddingInput parsed;
    if (item.is_string()) {
      if (item.str().empty())
        return BadRequest("'input' must not contain empty strings",
                          "empty_input");
      parsed.content.push_back({EmbeddingPart::Type::kText, item.str()});
    } else if (native && item.is_object() && item.contains("prompt_string")) {
      // llama.cpp's multimodal prompt: each marker in the string stands for
      // the next base64 file in `multimodal_data`.
      const auto* prompt = item.find("prompt_string");
      const auto* data = item.find("multimodal_data");
      if (!prompt->is_string() || item.size() != (data != nullptr ? 2U : 1U) ||
          (data != nullptr && !data->is_array()))
        return BadRequest(
            "prompt objects take a 'prompt_string' string and an optional "
            "'multimodal_data' array",
            "invalid_parameter_type");
      const std::string_view text = prompt->str();
      const std::size_t files = data != nullptr ? data->size() : 0;
      std::size_t markers = 0;
      for (std::size_t at = text.find(kMediaMarker);
           at != std::string_view::npos;
           at = text.find(kMediaMarker, at + kMediaMarker.size()))
        ++markers;
      if (markers != files)
        return BadRequest("'prompt_string' has " + std::to_string(markers) +
                              " media markers (" + std::string(kMediaMarker) +
                              ") but 'multimodal_data' has " +
                              std::to_string(files) + " items",
                          "invalid_multimodal_data");
      std::size_t begin = 0;
      for (std::size_t i = 0; i <= files; ++i) {
        const std::size_t end =
            i == files ? text.size() : text.find(kMediaMarker, begin);
        if (end > begin)
          parsed.content.push_back(
              {EmbeddingPart::Type::kText,
               std::string(text.substr(begin, end - begin))});
        if (i == files)
          break;
        const auto& file = data->items()[i];
        if (!file.is_string() || file.str().empty())
          return BadRequest("'multimodal_data' items must be base64 strings",
                            "invalid_parameter_type");
        if (LooksLikeAudio(file.str())) {
          if (++audio_count > EmbeddingService::kMaxAudio)
            return BadRequest(
                "embedding requests accept at most 32 audio clips",
                "too_many_audio");
          // The audio reader takes raw base64; the decoder checks the bytes.
          parsed.content.push_back(
              {EmbeddingPart::Type::kAudio, std::string(file.str())});
        } else {
          if (++image_count > EmbeddingService::kMaxImages)
            return BadRequest("embedding requests accept at most 32 images",
                              "too_many_images");
          // The image reader identifies PNG, JPEG and WebP by their bytes.
          parsed.content.push_back(
              {EmbeddingPart::Type::kImage,
               "data:image/jpeg;base64," + std::string(file.str())});
        }
        begin = end + kMediaMarker.size();
      }
      if (parsed.content.empty())
        return BadRequest("'prompt_string' must not be empty", "empty_input");
    } else if (item.is_object()) {
      const auto* content = item.find("content");
      if (item.size() != 1 || !content || !content->is_array())
        return BadRequest("typed input requires a 'content' array",
                          "invalid_parameter_type");
      if (content->items().empty())
        return BadRequest("'content' must not be empty", "empty_input");
      for (const auto& part : content->items()) {
        if (!part.is_object())
          return BadRequest("content parts must be objects",
                            "invalid_parameter_type");
        const auto type = part.member_str("type");
        if (type == "text") {
          const auto* text = part.find("text");
          if (part.size() != 2 || !text || !text->is_string())
            return BadRequest("text parts require a 'text' string",
                              "invalid_parameter_type");
          parsed.content.push_back({EmbeddingPart::Type::kText, text->str()});
        } else if (type == "image_url") {
          const auto* image = part.find("image_url");
          const auto* url =
              image && image->is_object() ? image->find("url") : nullptr;
          if (part.size() != 2 || !url || !url->is_string() ||
              url->str().empty() || image->size() != 1)
            return BadRequest("image parts require 'image_url': {'url': '...'}",
                              "invalid_parameter_type");
          if (++image_count > EmbeddingService::kMaxImages)
            return BadRequest("embedding requests accept at most 32 images",
                              "too_many_images");
          parsed.content.push_back({EmbeddingPart::Type::kImage, url->str()});
        } else if (type == "input_video") {
          const auto* video = part.find("input_video");
          if (part.size() != 2 || !video || !video->is_object())
            return BadRequest("video parts require an 'input_video' object",
                              "invalid_parameter_type");
          for (const auto& [key, value] : video->members()) {
            if ((key != "data" && key != "url") || !value.is_string())
              return BadRequest(
                  "input_video accepts string data and url fields",
                  "invalid_parameter_type");
          }
          const auto* source = video->find("data");
          if (!source)
            source = video->find("url");
          if (!source || source->str().empty())
            return BadRequest("input_video requires nonempty data or url",
                              "invalid_parameter_type");
          if (++video_count > EmbeddingService::kMaxVideos)
            return BadRequest("embedding requests accept at most 32 videos",
                              "too_many_videos");
          parsed.content.push_back(
              {EmbeddingPart::Type::kVideo, source->str()});
        } else if (type == "input_audio") {
          const auto* audio = part.find("input_audio");
          if (part.size() != 2 || !audio || !audio->is_object())
            return BadRequest("audio parts require an 'input_audio' object",
                              "invalid_parameter_type");
          for (const auto& [key, value] : audio->members()) {
            if ((key != "data" && key != "url" && key != "format") ||
                !value.is_string())
              return BadRequest(
                  "input_audio accepts string data, url and format fields",
                  "invalid_parameter_type");
          }
          // Match llama.cpp: data takes precedence; format is ignored and the
          // actual container is detected from its bytes.
          const auto* source = audio->find("data");
          if (!source)
            source = audio->find("url");
          if (!source || source->str().empty())
            return BadRequest("input_audio requires nonempty data or url",
                              "invalid_parameter_type");
          if (++audio_count > EmbeddingService::kMaxAudio)
            return BadRequest(
                "embedding requests accept at most 32 audio clips",
                "too_many_audio");
          parsed.content.push_back(
              {EmbeddingPart::Type::kAudio, source->str()});
        } else {
          return BadRequest("unsupported embedding content type: " + type,
                            "unsupported_input");
        }
      }
      if (std::ranges::none_of(parsed.content,
                               [](const auto& p) { return !p.value.empty(); }))
        return BadRequest("'content' must contain text or an image",
                          "empty_input");
    } else if (item.is_number() || item.is_array()) {
      return BadRequest("token-array input is not supported; send strings",
                        "unsupported_input");
    } else {
      return BadRequest("'input' items must be strings or content objects",
                        "invalid_parameter_type");
    }
    inputs.push_back(std::move(parsed));
    HttpResponse ok;
    ok.status = 200;
    return ok;
  };
  // llama.cpp also takes one bare prompt or content object.
  if (input->is_string() || (native && input->is_object())) {
    auto result = parse_item(*input);
    if (result.status != 200)
      return result;
  } else if (input->is_array()) {
    if (input->items().empty())
      return BadRequest("'input' must not be an empty array", "empty_input");
    if (input->size() > kMaximumInputs)
      return BadRequest("'input' accepts at most " +
                            std::to_string(kMaximumInputs) + " items",
                        "too_many_inputs");
    for (const auto& item : input->items()) {
      auto result = parse_item(item);
      if (result.status != 200)
        return result;
    }
  } else {
    return BadRequest(
        "'input' must be a string or an array of strings or content objects",
        "invalid_parameter_type");
  }

  bool base64 = false;
  if (const json::Value* format = Optional(root, "encoding_format");
      format != nullptr) {
    if (!format->is_string() ||
        (format->str() != "float" && format->str() != "base64")) {
      return BadRequest("'encoding_format' must be \"float\" or \"base64\"",
                        "invalid_encoding_format");
    }
    // As in llama.cpp, the native form validates it but returns floats.
    base64 = !native && format->str() == "base64";
  }
  // llama.cpp's per-request normalization. EmbeddingGemma 2 vectors are
  // always L2-normalized, so only its default, 2 (Euclidean), is accepted.
  if (const json::Value* normalize = Optional(root, "embd_normalize");
      normalize != nullptr) {
    if (!normalize->is_number() || normalize->as_double() != 2.0)
      return BadRequest("'embd_normalize' supports only 2 (Euclidean)",
                        "unsupported_embd_normalize");
  }

  std::size_t dimensions = 0;  // 0 means the model's full width.
  if (const json::Value* value = Optional(root, "dimensions");
      value != nullptr) {
    dimensions = value->as_size(0);
    if (std::ranges::find(kDimensions, dimensions) == kDimensions.end()) {
      return BadRequest("'dimensions' must be one of 128, 256, 512, or 768",
                        "invalid_dimensions");
    }
  }
  if (const json::Value* user = Optional(root, "user");
      user != nullptr && !user->is_string()) {
    return BadRequest("'user' must be a string", "invalid_parameter_type");
  }

  std::string prompt_name = service.default_prompt_name();
  bool requested_prompt = false;
  if (const json::Value* value = Optional(root, "prompt_name");
      value != nullptr) {
    if (!value->is_string()) {
      return BadRequest("'prompt_name' must be a string",
                        "invalid_parameter_type");
    }
    prompt_name = value->str();
    requested_prompt = true;
  }
  if (!prompt_name.empty()) {
    const std::string* prefix = service.prompt(prompt_name);
    if (prefix == nullptr) {
      return BadRequest("unknown prompt_name '" + prompt_name + "'",
                        "invalid_prompt_name");
    }
    // A leading text part: the service joins adjacent text before
    // tokenizing, so this equals a client-side prefix. A requested prompt
    // applies to every input, as in sentence-transformers. The model card
    // passes images and audio without a prefix, so the server default skips
    // inputs without text.
    for (EmbeddingInput& input : inputs) {
      const bool has_text =
          std::ranges::any_of(input.content, [](const EmbeddingPart& part) {
            return part.type == EmbeddingPart::Type::kText &&
                   !part.value.empty();
          });
      if (requested_prompt || has_text) {
        input.content.insert(input.content.begin(),
                             {EmbeddingPart::Type::kText, *prefix});
      }
    }
  }

  EmbeddingRequestOptions options;
  if (const auto* video = Optional(root, "video_options")) {
    if (!video->is_object())
      return BadRequest("video_options must be an object",
                        "invalid_parameter_type");
    for (const auto& [key, value] : video->members()) {
      if (key == "fps") {
        if (!value.is_number() || !std::isfinite(value.as_double()) ||
            value.as_double() <= 0 || value.as_double() > 120)
          return BadRequest("video fps must be in (0, 120]",
                            "invalid_video_options");
        options.video.fps = value.as_double();
      } else if (key == "max_frames") {
        const auto n = value.as_size(0);
        if (!value.is_number() || n < 1 || n > 32 ||
            value.as_double() != static_cast<double>(n))
          return BadRequest("video max_frames must be an integer in [1, 32]",
                            "invalid_video_options");
        options.video.max_frames = n;
      } else if (key == "add_timestamps") {
        if (!value.is_bool())
          return BadRequest("add_timestamps must be boolean",
                            "invalid_video_options");
        options.video.timestamps = value.as_bool();
      } else if (key == "overflow_strategy") {
        if (!value.is_string() ||
            (value.str() != "uniform" && value.str() != "truncate"))
          return BadRequest(
              "video overflow_strategy must be uniform or truncate",
              "invalid_video_options");
        options.video.truncate_overflow = value.str() == "truncate";
      } else
        return BadRequest("unsupported video option: " + key,
                          "invalid_video_options");
    }
  }

  if (const json::Value* truncate = Optional(root, "truncate");
      truncate != nullptr) {
    if (!truncate->is_bool()) {
      return BadRequest("'truncate' must be a boolean",
                        "invalid_parameter_type");
    }
    options.truncate = truncate->as_bool();
  }

  // vLLM's per-request processor options. Only the image budget is supported.
  if (const json::Value* kwargs = Optional(root, "mm_processor_kwargs");
      kwargs != nullptr) {
    if (!kwargs->is_object()) {
      return BadRequest("'mm_processor_kwargs' must be an object",
                        "invalid_parameter_type");
    }
    for (const auto& [key, value] : kwargs->members()) {
      if (key != "max_soft_tokens") {
        return BadRequest("unsupported mm_processor_kwargs key '" + key +
                              "'; only 'max_soft_tokens' is supported",
                          "invalid_mm_processor_kwargs");
      }
      const auto budget = value.as_size(0);
      if (budget > kMaxImageSoftTokens ||
          !(budget == 70 || budget == 140 || budget == 280 || budget == 560 ||
            budget == 1120)) {
        return BadRequest(
            "'max_soft_tokens' must be one of 70, 140, 280, 560, or 1120",
            "invalid_mm_processor_kwargs");
      }
      options.image_soft_tokens = static_cast<std::uint32_t>(budget);
    }
  }

  EmbeddingBatch batch;
  std::string error;
  switch (service.EmbedInputs(inputs, request.is_cancelled, &batch, &error,
                              options)) {
    case EmbeddingStatus::kOk:
      break;
    case EmbeddingStatus::kCancelled:
      return Error(499, "embedding request cancelled", "cancelled");
    case EmbeddingStatus::kInputTooLong:
      return BadRequest("input at index " + std::to_string(batch.failed_index) +
                            " exceeds the " +
                            std::to_string(EmbeddingService::kMaxInputTokens) +
                            "-token context window",
                        "context_length_exceeded");
    case EmbeddingStatus::kInvalidVideoInput:
      return BadRequest(error, "invalid_video_input");
    case EmbeddingStatus::kInvalidAudioInput:
      return BadRequest(error, "invalid_audio_input");
    case EmbeddingStatus::kInvalidInput:
      return BadRequest(
          error.empty() ? "invalid embedding input" : std::move(error),
          "invalid_image_input");
    case EmbeddingStatus::kFailed:
      return Error(500, error.empty() ? "embedding failed" : std::move(error),
                   "embedding_failed");
    case EmbeddingStatus::kDeviceLost:
      return Error(503, "GPU device lost; restart the server", "device_lost");
  }
  if (request.is_cancelled && request.is_cancelled()) {
    return Error(499, "embedding request cancelled", "cancelled");
  }

  if (batch.embeddings.size() != inputs.size() ||
      batch.token_counts.size() != inputs.size()) {
    return Error(500, "embedding runtime returned an inconsistent batch",
                 "embedding_failed");
  }
  std::size_t prompt_tokens = 0;
  for (const std::size_t count : batch.token_counts) {
    prompt_tokens += count;
  }

  // llama.cpp's native form is a bare array whose items nest the pooled
  // vector in a one-element list, without model or usage.
  std::string body = native ? "[" : R"({"object":"list","data":[)";
  std::vector<float> truncated;
  for (std::size_t index = 0; index < batch.embeddings.size(); ++index) {
    std::span<const float> values = batch.embeddings[index];
    if (values.empty()) {
      return Error(500, "embedding runtime returned an empty embedding",
                   "embedding_failed");
    }
    for (const float value : values) {
      if (!std::isfinite(value)) {
        return Error(500, "embedding runtime returned a non-finite value",
                     "embedding_failed");
      }
    }
    if (dimensions != 0 && dimensions < values.size()) {
      if (!TruncateAndNormalize(values, dimensions, &truncated)) {
        return Error(500, "embedding cannot be truncated to the requested size",
                     "embedding_failed");
      }
      values = truncated;
    }
    if (index != 0) {
      body += ',';
    }
    if (native) {
      body += R"({"index":)" + std::to_string(index) + R"(,"embedding":[)";
      AppendFloats(&body, values);
      body += "]}";
      continue;
    }
    body += R"({"object":"embedding","index":)" + std::to_string(index) +
            R"(,"embedding":)";
    if (base64) {
      body += '"' + EncodeBase64Floats(values) + '"';
    } else {
      AppendFloats(&body, values);
    }
    body += '}';
  }
  if (native)
    body += ']';
  else
    body += R"(],"model":)" + json::Value(service.model_id()).dump() +
            R"(,"usage":{"prompt_tokens":)" + std::to_string(prompt_tokens) +
            R"(,"total_tokens":)" + std::to_string(prompt_tokens) + "}}";

  HttpResponse response;
  response.status = 200;
  response.reason = "OK";
  response.body = std::move(body);
  response.headers = {
      {"Content-Type", "application/json"},
      {"X-Gufo-Schema", std::string(kEmbeddingsApiSchema)},
      {"X-Gufo-Embedding-Backend", service.backend_name()},
  };
  response.log_details =
      "model=" + service.model_id() +
      " inputs=" + std::to_string(inputs.size()) +
      " prompt_tokens=" + std::to_string(prompt_tokens) + " dimensions=" +
      std::to_string(dimensions != 0 ? dimensions
                                     : batch.embeddings.front().size()) +
      " encoding_format=" + (base64 ? "base64" : "float") +
      (native ? " shape=llama.cpp" : "");
  return response;
}

}  // namespace

bool IsEmbeddingsApiPath(std::string_view path) noexcept {
  return path == kEmbeddingsPath || path == kNativeEmbeddingsPath ||
         path == kLegacyEmbeddingPath;
}

HttpResponse HandleEmbeddingsApiRequest(const HttpRequest& request,
                                        EmbeddingService& service) {
  if (IsEmbeddingsApiPath(request.path) && request.method == "POST") {
    return Embed(request, service, request.path != kEmbeddingsPath);
  }
  return Error(405, "HTTP method is not supported for this embeddings endpoint",
               "method_not_allowed");
}

}  // namespace gufo::server
