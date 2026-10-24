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

bool IsKnownField(std::string_view name) {
  return name == "input" || name == "model" || name == "encoding_format" ||
         name == "dimensions" || name == "user";
}

/// The member's value, or nullptr when it is absent or null.
const json::Value* Optional(const json::Value& root, const std::string& name) {
  const json::Value* value = root.find(name);
  return value == nullptr || value->is_null() ? nullptr : value;
}

HttpResponse Embed(const HttpRequest& request, EmbeddingService& service) {
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
    if (!IsKnownField(name)) {
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
  if (input == nullptr) {
    return BadRequest("'input' is required", "missing_input");
  }
  std::vector<EmbeddingInput> inputs;
  std::size_t image_count = 0;
  const auto parse_item = [&](const json::Value& item) -> HttpResponse {
    EmbeddingInput parsed;
    if (item.is_string()) {
      if (item.str().empty())
        return BadRequest("'input' must not contain empty strings",
                          "empty_input");
      parsed.content.push_back({EmbeddingPart::Type::kText, item.str()});
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
  if (input->is_string()) {
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
    base64 = format->str() == "base64";
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

  EmbeddingBatch batch;
  std::string error;
  switch (service.EmbedInputs(inputs, request.is_cancelled, &batch, &error)) {
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

  std::string body = R"({"object":"list","data":[)";
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
    body += R"({"object":"embedding","index":)" + std::to_string(index) +
            R"(,"embedding":)";
    if (base64) {
      body += '"' + EncodeBase64Floats(values) + '"';
    } else {
      AppendFloats(&body, values);
    }
    body += '}';
  }
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
      " encoding_format=" + (base64 ? "base64" : "float");
  return response;
}

}  // namespace

bool IsEmbeddingsApiPath(std::string_view path) noexcept {
  return path == kEmbeddingsPath;
}

HttpResponse HandleEmbeddingsApiRequest(const HttpRequest& request,
                                        EmbeddingService& service) {
  if (request.path == kEmbeddingsPath && request.method == "POST") {
    return Embed(request, service);
  }
  return Error(405, "HTTP method is not supported for this embeddings endpoint",
               "method_not_allowed");
}

}  // namespace gufo::server
