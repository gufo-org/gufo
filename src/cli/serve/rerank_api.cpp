#include "src/cli/serve/rerank_api.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <exception>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::server {
namespace {

using ext::EmbeddingPart;

// llama-server's three routes share one handler there.
constexpr std::string_view kRerankPath = "/v1/rerank";
constexpr std::string_view kRerankingPath = "/v1/reranking";
constexpr std::string_view kNativeRerankPath = "/rerank";

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

/// The member's value, or nullptr when it is absent or null.
const json::Value* Optional(const json::Value& root, const std::string& name) {
  const json::Value* value = root.find(name);
  return value == nullptr || value->is_null() ? nullptr : value;
}

/// Shortest decimal text that parses back to the same float.
std::string FloatText(float value) {
  char buffer[32];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
  return std::string(buffer, result.ptr);
}

/// The parts of a `{"content": [...]}` query or document: the text,
/// `image_url` and `input_video` parts `/v1/embeddings` takes, in order.
bool PartsOf(const json::Value& content, const std::string& what,
             RerankInput* input, HttpResponse* error) {
  if (!content.is_array() || content.items().empty()) {
    *error = BadRequest(what + " 'content' must be a non-empty array",
                        "invalid_parameter_type");
    return false;
  }
  for (const auto& part : content.items()) {
    const auto type = part.is_object() ? part.member_str("type") : "";
    if (type == "text") {
      const auto* text = part.find("text");
      if (part.size() != 2 || !text || !text->is_string()) {
        *error = BadRequest("text parts require a 'text' string",
                            "invalid_parameter_type");
        return false;
      }
      input->content.push_back({EmbeddingPart::Type::kText, text->str()});
    } else if (type == "image_url") {
      const auto* image = part.find("image_url");
      const auto* url =
          image && image->is_object() ? image->find("url") : nullptr;
      if (part.size() != 2 || !url || !url->is_string() || url->str().empty() ||
          image->size() != 1) {
        *error = BadRequest("image parts require 'image_url': {'url': '...'}",
                            "invalid_parameter_type");
        return false;
      }
      input->content.push_back({EmbeddingPart::Type::kImage, url->str()});
    } else if (type == "input_video") {
      const auto* video = part.find("input_video");
      bool valid = part.size() == 2 && video && video->is_object();
      const json::Value* source = nullptr;
      if (valid) {
        for (const auto& [key, value] : video->members()) {
          valid = valid && (key == "data" || key == "url") && value.is_string();
        }
        source = video->find("data");
        if (!source) {
          source = video->find("url");
        }
      }
      if (!valid || !source || source->str().empty()) {
        *error = BadRequest(
            "video parts require 'input_video': {'data' or 'url': '...'}",
            "invalid_parameter_type");
        return false;
      }
      input->content.push_back({EmbeddingPart::Type::kVideo, source->str()});
    } else if (part.is_object()) {
      *error = BadRequest("unsupported rerank content type: " + type,
                          "unsupported_input");
      return false;
    } else {
      *error =
          BadRequest("content parts must be objects", "invalid_parameter_type");
      return false;
    }
  }
  if (std::ranges::none_of(input->content, [](const auto& part) {
        return !part.value.empty();
      })) {
    *error = BadRequest(what + " 'content' must not be empty", "empty_input");
    return false;
  }
  return true;
}

/// A query or document: a string, llama.cpp's `{"text": ...}` object, or a
/// `{"content": [...]}` object of parts (see `PartsOf`). Any other member,
/// such as the `image` member of some rerankers' own APIs, is rejected.
/// Returns false with `*error` set otherwise.
bool InputOf(const json::Value& value, const std::string& what,
             RerankInput* input, HttpResponse* error) {
  if (value.is_string()) {
    input->content.push_back({EmbeddingPart::Type::kText, value.str()});
    return true;
  }
  if (value.is_object()) {
    if (const json::Value* content = Optional(value, "content")) {
      for (const auto& [key, member] : value.members()) {
        if (key != "content" && !member.is_null()) {
          *error = BadRequest(what + " takes 'text' or 'content', not both",
                              "invalid_parameter_type");
          return false;
        }
      }
      return PartsOf(*content, what, input, error);
    }
    for (const auto& [key, member] : value.members()) {
      if (key == "text" || member.is_null()) {
        continue;
      }
      *error = BadRequest(
          what + " member '" + key +
              "' is not supported; send images and videos as 'content' parts",
          "unsupported_input");
      return false;
    }
    const json::Value* inner = Optional(value, "text");
    if (inner != nullptr && inner->is_string()) {
      input->content.push_back({EmbeddingPart::Type::kText, inner->str()});
      return true;
    }
  }
  *error = BadRequest(what +
                          " must be a string or an object with a 'text' "
                          "string or a 'content' array",
                      "invalid_parameter_type");
  return false;
}

/// The text parts of `input`, joined, as TEI's `return_text` returns them.
std::string TextOf(const RerankInput& input) {
  std::string text;
  for (const EmbeddingPart& part : input.content) {
    if (part.type == EmbeddingPart::Type::kText) {
      text += part.value;
    }
  }
  return text;
}

HttpResponse Rerank(const HttpRequest& request, RerankService& service) {
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
  // As in llama.cpp, unknown members are ignored.

  if (const json::Value* model = Optional(root, "model"); model != nullptr) {
    if (!model->is_string()) {
      return BadRequest("'model' must be a string", "invalid_parameter_type");
    }
    if (!service.serves_model(model->str())) {
      return Error(404, "requested rerank model is not loaded",
                   "model_not_found");
    }
  }

  // llama.cpp tells the TEI form by the presence of `texts`, whatever else the
  // body holds.
  const bool tei = root.contains("texts");

  const json::Value* query_value = Optional(root, "query");
  if (query_value == nullptr) {
    return BadRequest("\"query\" must be provided", "missing_query");
  }
  RerankInput query;
  HttpResponse problem;
  if (!InputOf(*query_value, "'query'", &query, &problem)) {
    return problem;
  }
  if (std::ranges::none_of(query.content, [](const auto& part) {
        return !part.value.empty();
      })) {
    return BadRequest("'query' must not be empty", "empty_query");
  }

  const json::Value* documents_value = Optional(root, "documents");
  if (documents_value == nullptr) {
    documents_value = Optional(root, "texts");
  }
  if (documents_value == nullptr || !documents_value->is_array() ||
      documents_value->size() == 0) {
    return BadRequest(
        "\"documents\" must be a non-empty array of strings or objects",
        "invalid_documents");
  }
  if (documents_value->size() > RerankService::kMaxDocuments) {
    return BadRequest("a request takes at most " +
                          std::to_string(RerankService::kMaxDocuments) +
                          " documents",
                      "too_many_documents");
  }
  std::vector<RerankInput> documents;
  documents.reserve(documents_value->size());
  for (const auto& item : documents_value->items()) {
    RerankInput document;
    if (!InputOf(item, "each document", &document, &problem)) {
      return problem;
    }
    documents.push_back(std::move(document));
  }

  std::size_t top_n = documents.size();
  if (const json::Value* value = Optional(root, "top_n"); value != nullptr) {
    if (!value->is_number() ||
        value->as_double() != std::floor(value->as_double())) {
      return BadRequest("\"top_n\" must be an integer",
                        "invalid_parameter_type");
    }
    if (value->as_double() < 0) {
      return BadRequest("\"top_n\" must be >= 0", "invalid_top_n");
    }
    top_n = std::min(top_n, value->as_size(documents.size()));
  }

  // `return_text` belongs to the TEI form only, as in llama.cpp.
  bool return_text = false;
  if (const json::Value* value = Optional(root, "return_text");
      value != nullptr && tei) {
    if (!value->is_bool()) {
      return BadRequest("'return_text' must be a boolean",
                        "invalid_parameter_type");
    }
    return_text = value->as_bool();
  }

  // Gufo's own members: the instruction of an instruction-following reranker,
  // and TEI's `truncate`.
  RerankRequestOptions options;
  if (const json::Value* value = Optional(root, "instruction");
      value != nullptr) {
    if (!value->is_string()) {
      return BadRequest("'instruction' must be a string",
                        "invalid_parameter_type");
    }
    options.instruction = value->str();
  }
  if (const json::Value* value = Optional(root, "truncate"); value != nullptr) {
    if (!value->is_bool()) {
      return BadRequest("'truncate' must be a boolean",
                        "invalid_parameter_type");
    }
    options.truncate = value->as_bool();
  }

  RerankScores scores;
  std::string error;
  switch (service.Rerank(query, documents, request.is_cancelled, &scores,
                         &error, options)) {
    case RerankStatus::kOk:
      break;
    case RerankStatus::kCancelled:
      return Error(499, "rerank request cancelled", "cancelled");
    case RerankStatus::kInputTooLong:
      return BadRequest(scores.failed_index < documents.size()
                            ? "document at index " +
                                  std::to_string(scores.failed_index) +
                                  " exceeds the model's context window"
                            : "a document exceeds the model's context window",
                        "context_length_exceeded");
    case RerankStatus::kInvalidInput:
      return BadRequest(
          error.empty() ? "invalid rerank input" : std::move(error),
          "invalid_input");
    case RerankStatus::kUnsupportedInput:
      return BadRequest(
          error.empty() ? "unsupported rerank input" : std::move(error),
          "unsupported_input");
    case RerankStatus::kFailed:
      return Error(500, error.empty() ? "rerank failed" : std::move(error),
                   "rerank_failed");
    case RerankStatus::kDeviceLost:
      return Error(503, "GPU device lost; restart the server", "device_lost");
  }

  std::size_t prompt_tokens = 0;
  for (const std::size_t count : scores.token_counts) {
    prompt_tokens += count;
  }

  // Highest score first; equal scores keep their input order.
  std::vector<std::size_t> order(documents.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t left, std::size_t right) {
                     return scores.scores[left] > scores.scores[right];
                   });
  order.resize(top_n);

  std::string body =
      tei ? "["
          : R"({"model":)" + json::Value(service.model_id()).dump() +
                R"(,"object":"list","usage":{)" + R"("prompt_tokens":)" +
                std::to_string(prompt_tokens) + R"(,"total_tokens":)" +
                std::to_string(prompt_tokens) + R"(},"results":[)";
  for (std::size_t rank = 0; rank < order.size(); ++rank) {
    const std::size_t index = order[rank];
    if (rank != 0) {
      body += ',';
    }
    body += R"({"index":)" + std::to_string(index) +
            (tei ? R"(,"score":)" : R"(,"relevance_score":)") +
            FloatText(scores.scores[index]);
    if (tei && return_text) {
      body += R"(,"text":)" + json::Value(TextOf(documents[index])).dump();
    }
    body += '}';
  }
  body += tei ? "]" : "]}";

  HttpResponse response;
  response.status = 200;
  response.reason = "OK";
  response.body = std::move(body);
  response.headers = {
      {"Content-Type", "application/json"},
      {"X-Gufo-Schema", std::string(kRerankApiSchema)},
      {"X-Gufo-Rerank-Backend", service.backend_name()},
  };
  response.log_details = "model=" + service.model_id() +
                         " documents=" + std::to_string(documents.size()) +
                         " results=" + std::to_string(order.size()) +
                         " prompt_tokens=" + std::to_string(prompt_tokens) +
                         (tei ? " shape=tei" : " shape=jina");
  return response;
}

}  // namespace

bool IsRerankApiPath(std::string_view path) noexcept {
  return path == kRerankPath || path == kRerankingPath ||
         path == kNativeRerankPath;
}

HttpResponse HandleRerankApiRequest(const HttpRequest& request,
                                    RerankService& service) {
  if (IsRerankApiPath(request.path) && request.method == "POST") {
    return Rerank(request, service);
  }
  return Error(405, "HTTP method is not supported for this rerank endpoint",
               "method_not_allowed");
}

}  // namespace gufo::server
