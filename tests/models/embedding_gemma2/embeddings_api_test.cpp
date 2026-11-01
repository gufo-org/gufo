// POST /v1/embeddings contract against a fake encoder: no model files, no
// device, and no socket. The test calls the HTTP handler directly.
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "tests/support/model_api_test_support.hpp"

namespace {

using gufo::server::EmbeddingBatch;
using gufo::server::EmbeddingInput;
using gufo::server::EmbeddingPart;
using gufo::server::EmbeddingRequestOptions;
using gufo::server::EmbeddingService;
using gufo::server::EmbeddingServiceOptions;
using gufo::server::EmbeddingStatus;
using gufo::server::HandleEmbeddingsApiRequest;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;
using gufo::server::IsEmbeddingsApiPath;
using gufo::server::LoadSentenceTransformerPrompts;
using gufo::server::TruncateKeepingImages;

constexpr std::size_t kWidth = 768;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL embedding_gemma2_embeddings_api_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

/// A deterministic unit vector per text. Components differ in magnitude so a
/// truncated prefix is not already normalized.
std::vector<float> FakeEmbedding(const std::string& text) {
  std::vector<float> vector(kWidth);
  const std::size_t seed = std::hash<std::string>{}(text);
  double sum = 0.0;
  for (std::size_t i = 0; i < kWidth; ++i) {
    const double value =
        std::sin(static_cast<double>(seed % 1009U) + 0.37 * (double)i) *
        (1.0 + static_cast<double>(i % 7U));
    vector[i] = static_cast<float>(value);
    sum += static_cast<double>(vector[i]) * static_cast<double>(vector[i]);
  }
  const double scale = 1.0 / std::sqrt(sum);
  for (float& value : vector) {
    value = static_cast<float>(static_cast<double>(value) * scale);
  }
  return vector;
}

std::size_t FakeTokens(const std::string& text) {
  return text.size() + 2U;  // One token per byte plus <bos> and <eos>.
}

struct FakeRunner {
  EmbeddingStatus Run(const std::vector<std::string>& texts,
                      const EmbeddingRequestOptions& request,
                      EmbeddingBatch* batch, std::string* error) {
    ++calls;
    last_texts = texts;
    last_truncate = request.truncate;
    last_image_soft_tokens = request.image_soft_tokens;
    for (std::size_t i = 0; i < texts.size(); ++i) {
      if (texts[i].starts_with("TOOLONG")) {
        if (request.truncate) {
          batch->embeddings.push_back(FakeEmbedding(texts[i]));
          batch->token_counts.push_back(EmbeddingService::kMaxInputTokens);
          continue;
        }
        batch->failed_index = i;
        return EmbeddingStatus::kInputTooLong;
      }
      if (texts[i] == "BROKEN") {
        *error = "device exploded";
        return EmbeddingStatus::kFailed;
      }
      if (texts[i] == "LOST") {
        *error = "device reset";
        return EmbeddingStatus::kDeviceLost;
      }
      batch->embeddings.push_back(FakeEmbedding(texts[i]));
      batch->token_counts.push_back(FakeTokens(texts[i]));
    }
    return EmbeddingStatus::kOk;
  }

  int calls{0};
  std::vector<std::string> last_texts;
  bool last_truncate{false};
  std::uint32_t last_image_soft_tokens{0};
};

std::unique_ptr<EmbeddingService> MakeService(FakeRunner* runner) {
  return std::make_unique<EmbeddingService>(EmbeddingServiceOptions{
      .model_root = {},
      .validate_model = false,
      .model_id = "embeddinggemma-2",
      .prompts = {},
      .default_prompt_name = {},
      .runner =
          [runner](const std::vector<EmbeddingInput>& inputs,
                   const EmbeddingRequestOptions& request,
                   EmbeddingBatch* batch, std::string* error) {
            std::vector<std::string> texts;
            for (const auto& input : inputs) {
              std::string text;
              for (const auto& part : input.content)
                text += part.value;
              texts.push_back(std::move(text));
            }
            return runner->Run(texts, request, batch, error);
          },
  });
}

HttpResponse Send(EmbeddingService& service, std::string body,
                  std::string method = "POST",
                  std::function<bool()> cancelled = {}) {
  return HandleEmbeddingsApiRequest(
      HttpRequest{
          .method = std::move(method),
          .path = "/v1/embeddings",
          .query = {},
          .body = std::move(body),
          .headers = {{"Content-Type", "application/json"}},
          .is_cancelled = std::move(cancelled),
      },
      service);
}

HttpResponse SendTo(EmbeddingService& service, std::string path,
                    std::string body) {
  return HandleEmbeddingsApiRequest(
      HttpRequest{
          .method = "POST",
          .path = std::move(path),
          .query = {},
          .body = std::move(body),
          .headers = {{"Content-Type", "application/json"}},
          .is_cancelled = {},
      },
      service);
}

gufo::json::Value Parse(const HttpResponse& response) {
  try {
    return gufo::json::parse(response.body);
  } catch (const std::exception& error) {
    Fail(std::string("response is not JSON: ") + error.what() + ": " +
         response.body);
  }
}

std::string ErrorCode(const HttpResponse& response) {
  const gufo::json::Value body = Parse(response);
  const gufo::json::Value* error = body.find("error");
  Check(error != nullptr && error->is_object(), "error object present");
  Check(error->contains("message") && error->contains("type"),
        "error carries message and type");
  return error->member_str("code");
}

void ExpectError(const HttpResponse& response, int status,
                 std::string_view code, const std::string& what) {
  Check(response.status == status, what + ": status " +
                                       std::to_string(response.status) +
                                       " body " + response.body);
  Check(ErrorCode(response) == code, what + ": code " + ErrorCode(response) +
                                         " expected " + std::string(code));
}

std::vector<float> FloatsOf(const gufo::json::Value& embedding) {
  Check(embedding.is_array(), "embedding is an array");
  std::vector<float> values;
  for (const auto& item : embedding.items()) {
    Check(item.is_number(), "embedding component is a number");
    values.push_back(static_cast<float>(item.as_double()));
  }
  return values;
}

double Norm(const std::vector<float>& values) {
  double sum = 0.0;
  for (const float value : values) {
    sum += static_cast<double>(value) * static_cast<double>(value);
  }
  return std::sqrt(sum);
}

std::vector<std::uint8_t> DecodeBase64(std::string_view text) {
  const auto value = [](char c) -> int {
    if (c >= 'A' && c <= 'Z')
      return c - 'A';
    if (c >= 'a' && c <= 'z')
      return c - 'a' + 26;
    if (c >= '0' && c <= '9')
      return c - '0' + 52;
    if (c == '+')
      return 62;
    if (c == '/')
      return 63;
    return -1;
  };
  Check(text.size() % 4U == 0U, "base64 length is a multiple of four");
  std::vector<std::uint8_t> bytes;
  for (std::size_t i = 0; i < text.size(); i += 4U) {
    std::uint32_t word = 0;
    int padding = 0;
    for (std::size_t j = 0; j < 4U; ++j) {
      const char c = text[i + j];
      word <<= 6U;
      if (c == '=') {
        ++padding;
      } else {
        const int decoded = value(c);
        Check(decoded >= 0 && padding == 0, "valid base64 character");
        word |= static_cast<std::uint32_t>(decoded);
      }
    }
    bytes.push_back(static_cast<std::uint8_t>(word >> 16U));
    if (padding < 2)
      bytes.push_back(static_cast<std::uint8_t>(word >> 8U));
    if (padding < 1)
      bytes.push_back(static_cast<std::uint8_t>(word));
  }
  return bytes;
}

void TestStringInput() {
  FakeRunner runner;
  auto service = MakeService(&runner);
  const HttpResponse response =
      Send(*service, R"({"model":"embeddinggemma-2","input":"hello world"})");
  Check(response.status == 200, "string input succeeds: " + response.body);
  Check(std::ranges::any_of(response.headers,
                            [](const auto& header) {
                              return header.first == "Content-Type" &&
                                     header.second == "application/json";
                            }),
        "response is JSON");
  const gufo::json::Value body = Parse(response);
  Check(body.member_str("object") == "list", "list object");
  Check(body.member_str("model") == "embeddinggemma-2", "model echoes id");
  Check(body.find("data")->items().size() == 1U, "one embedding");
  const gufo::json::Value& entry = body.find("data")->items().front();
  Check(entry.member_str("object") == "embedding", "embedding object");
  Check(entry.member_size("index", 99) == 0U, "index zero");
  const std::vector<float> values = FloatsOf(*entry.find("embedding"));
  Check(values == FakeEmbedding("hello world"),
        "floats round-trip exactly through the JSON text");
  Check(std::fabs(Norm(values) - 1.0) < 1e-5, "full vector is unit length");
  const gufo::json::Value& usage = *body.find("usage");
  Check(usage.member_size("prompt_tokens") == FakeTokens("hello world") &&
            usage.member_size("total_tokens") == FakeTokens("hello world"),
        "usage counts the markers");
  Check(runner.last_texts == std::vector<std::string>{"hello world"},
        "service receives the text unchanged");
}

void TestArrayInputAndOrder() {
  FakeRunner runner;
  auto service = MakeService(&runner);
  const std::vector<std::string> inputs = {"alpha", "b", "gamma gamma",
                                           "δελτα \"quoted\"\n"};
  gufo::json::Value input = gufo::json::Value::array();
  for (const auto& text : inputs) {
    input.push_back(text);
  }
  gufo::json::Value request = gufo::json::Value::object();
  request["input"] = std::move(input);  // Omitting `model` is accepted.
  request["user"] = "someone";
  const HttpResponse response = Send(*service, request.dump());
  Check(response.status == 200, "array input succeeds: " + response.body);
  Check(runner.last_texts == inputs, "texts reach the service in order");
  const gufo::json::Value body = Parse(response);
  const auto& data = body.find("data")->items();
  Check(data.size() == inputs.size(), "one embedding per input");
  std::size_t tokens = 0;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    Check(data[i].member_size("index", 99) == i, "index follows input order");
    Check(FloatsOf(*data[i].find("embedding")) == FakeEmbedding(inputs[i]),
          "embedding " + std::to_string(i) + " belongs to its input");
    tokens += FakeTokens(inputs[i]);
  }
  Check(body.find("usage")->member_size("prompt_tokens") == tokens &&
            body.find("usage")->member_size("total_tokens") == tokens,
        "usage sums every input");
}

void TestDimensions() {
  FakeRunner runner;
  auto service = MakeService(&runner);
  const std::vector<float> full = FakeEmbedding("matryoshka");
  for (const std::size_t dimensions : {128U, 256U, 512U, 768U}) {
    const HttpResponse response =
        Send(*service, R"({"input":"matryoshka","dimensions":)" +
                           std::to_string(dimensions) + "}");
    Check(response.status == 200, "dimensions accepted: " + response.body);
    const std::vector<float> values = FloatsOf(
        *Parse(response).find("data")->items().front().find("embedding"));
    Check(values.size() == dimensions, "dimension count honored");
    Check(std::fabs(Norm(values) - 1.0) < 1e-5,
          "truncated vector is renormalized");
    if (dimensions == kWidth) {
      Check(values == full, "768 leaves the vector untouched");
      continue;
    }
    // The truncation is a scaled prefix of the full vector.
    const double scale = static_cast<double>(values[0]) / full[0];
    Check(scale > 1.0, "a prefix of a unit vector scales up");
    for (std::size_t i = 0; i < dimensions; ++i) {
      Check(std::fabs(static_cast<double>(values[i]) - full[i] * scale) < 1e-5,
            "truncation keeps the leading components");
    }
  }
  for (const char* bad :
       {"100", "0", "1024", "-128", "128.5", "\"256\"", "true", "[128]"}) {
    ExpectError(Send(*service,
                     std::string(R"({"input":"x","dimensions":)") + bad + "}"),
                400, "invalid_dimensions", std::string("dimensions ") + bad);
  }
  // null means unspecified, as several client libraries send it.
  Check(Send(*service, R"({"input":"x","dimensions":null})").status == 200,
        "null dimensions is the default");
}

void TestBase64() {
  FakeRunner runner;
  auto service = MakeService(&runner);
  const HttpResponse response = Send(
      *service,
      R"({"input":["one","two"],"encoding_format":"base64","dimensions":256})");
  Check(response.status == 200, "base64 succeeds: " + response.body);
  const gufo::json::Value body = Parse(response);
  const auto& data = body.find("data")->items();
  Check(data.size() == 2U, "two embeddings");
  const std::vector<std::string> inputs = {"one", "two"};
  for (std::size_t i = 0; i < data.size(); ++i) {
    const gufo::json::Value* embedding = data[i].find("embedding");
    Check(embedding != nullptr && embedding->is_string(),
          "base64 embedding is a string");
    const std::vector<std::uint8_t> bytes = DecodeBase64(embedding->str());
    Check(bytes.size() == 256U * sizeof(float), "256 little-endian floats");
    std::vector<float> decoded;
    for (std::size_t offset = 0; offset < bytes.size(); offset += 4U) {
      const std::uint32_t bits = std::uint32_t{bytes[offset]} |
                                 (std::uint32_t{bytes[offset + 1U]} << 8U) |
                                 (std::uint32_t{bytes[offset + 2U]} << 16U) |
                                 (std::uint32_t{bytes[offset + 3U]} << 24U);
      decoded.push_back(std::bit_cast<float>(bits));
    }
    const std::vector<float> full = FakeEmbedding(inputs[i]);
    Check(std::fabs(Norm(decoded) - 1.0) < 1e-5, "decoded vector is unit");
    const double scale = static_cast<double>(decoded[0]) / full[0];
    for (std::size_t j = 0; j < decoded.size(); ++j) {
      Check(std::fabs(static_cast<double>(decoded[j]) - full[j] * scale) < 1e-5,
            "base64 carries the truncated components");
    }
  }
  // The same request in float form must agree bit for bit.
  const HttpResponse floats = Send(
      *service,
      R"({"input":["one","two"],"encoding_format":"float","dimensions":256})");
  const std::vector<float> from_float =
      FloatsOf(*Parse(floats).find("data")->items()[0].find("embedding"));
  const std::vector<std::uint8_t> bytes =
      DecodeBase64(data[0].find("embedding")->str());
  Check(std::memcmp(from_float.data(), bytes.data(), bytes.size()) == 0,
        "float and base64 encodings are the same numbers");
  ExpectError(Send(*service, R"({"input":"x","encoding_format":"hex"})"), 400,
              "invalid_encoding_format", "unknown encoding_format");
  ExpectError(Send(*service, R"({"input":"x","encoding_format":1})"), 400,
              "invalid_encoding_format", "non-string encoding_format");
}

void TestInputValidation() {
  FakeRunner runner;
  auto service = MakeService(&runner);
  ExpectError(Send(*service, R"({"input":""})"), 400, "empty_input",
              "empty string");
  ExpectError(Send(*service, R"({"input":[]})"), 400, "empty_input",
              "empty array");
  ExpectError(Send(*service, R"({"input":["fine",""]})"), 400, "empty_input",
              "empty array item");
  ExpectError(Send(*service, R"({"model":"embeddinggemma-2"})"), 400,
              "missing_input", "missing input");
  ExpectError(Send(*service, R"({"input":null})"), 400, "missing_input",
              "null input");
  const HttpResponse tokens = Send(*service, R"({"input":[1,2,3]})");
  ExpectError(tokens, 400, "unsupported_input", "token array");
  Check(Parse(tokens).find("error")->member_str("message") ==
            "token-array input is not supported; send strings",
        "token-array message is explicit");
  ExpectError(Send(*service, R"({"input":[[1,2],[3]]})"), 400,
              "unsupported_input", "nested token arrays");
  ExpectError(Send(*service, R"({"input":["text",7]})"), 400,
              "unsupported_input", "mixed array");
  ExpectError(Send(*service, R"({"input":{"a":1}})"), 400,
              "invalid_parameter_type", "object input");
  ExpectError(Send(*service, R"({"input":[true]})"), 400,
              "invalid_parameter_type", "boolean item");
  ExpectError(Send(*service, R"({"input":"x","user":3})"), 400,
              "invalid_parameter_type", "non-string user");
  ExpectError(Send(*service, R"({"input":"x","normalize":true})"), 400,
              "unsupported_field", "unknown member");

  std::string many = R"({"input":[)";
  for (int i = 0; i < 2049; ++i) {
    many += i == 0 ? "\"a\"" : ",\"a\"";
  }
  many += "]}";
  ExpectError(Send(*service, many), 400, "too_many_inputs", "2049 inputs");
  std::string limit = R"({"input":[)";
  for (int i = 0; i < 2048; ++i) {
    limit += i == 0 ? "\"a\"" : ",\"a\"";
  }
  limit += "]}";
  const HttpResponse accepted = Send(*service, limit);
  Check(accepted.status == 200 &&
            Parse(accepted).find("data")->items().size() == 2048U,
        "2048 inputs are accepted");
  Check(runner.calls == 1, "rejected requests never reach the encoder");
}

void TestModelAndJson() {
  FakeRunner runner;
  auto service = MakeService(&runner);
  ExpectError(Send(*service, R"({"model":"other","input":"x"})"), 404,
              "model_not_found", "wrong model");
  ExpectError(Send(*service, R"({"model":7,"input":"x"})"), 400,
              "invalid_parameter_type", "non-string model");
  ExpectError(Send(*service, ""), 400, "parse_error", "empty body");
  ExpectError(Send(*service, "{\"input\":"), 400, "parse_error",
              "truncated JSON");
  ExpectError(Send(*service, "not json"), 400, "parse_error", "not JSON");
  ExpectError(Send(*service, R"(["x"])"), 400, "invalid_request_body",
              "array body");
  ExpectError(Send(*service, R"("x")"), 400, "invalid_request_body",
              "string body");
  ExpectError(Send(*service, R"({"input":"x"})", "GET"), 405,
              "method_not_allowed", "GET");
  Check(runner.calls == 0, "invalid requests never reach the encoder");
  Check(IsEmbeddingsApiPath("/v1/embeddings") &&
            !IsEmbeddingsApiPath("/v1/embeddings/") &&
            !IsEmbeddingsApiPath("/v1/chat/completions"),
        "path predicate is exact");
}

void TestLimitsAndFailures() {
  FakeRunner runner;
  auto service = MakeService(&runner);
  const HttpResponse long_input =
      Send(*service, R"({"input":["short","TOOLONG input"]})");
  ExpectError(long_input, 400, "context_length_exceeded", "over-length input");
  const std::string message =
      Parse(long_input).find("error")->member_str("message");
  Check(message.find("index 1") != std::string::npos &&
            message.find("8192") != std::string::npos,
        "message names the input index and the limit: " + message);

  const HttpResponse broken = Send(*service, R"({"input":"BROKEN"})");
  ExpectError(broken, 500, "embedding_failed", "runtime failure");
  Check(Parse(broken).find("error")->member_str("message") == "device exploded",
        "runtime error text is returned");
  Check(Parse(broken).find("error")->member_str("type") == "server_error",
        "runtime failures are server errors");
  Check(!service->device_lost(), "an ordinary failure keeps the device");

  const HttpResponse lost = Send(*service, R"({"input":"LOST"})");
  ExpectError(lost, 503, "device_lost", "lost device");
  Check(Parse(lost).find("error")->member_str("type") == "server_error",
        "a lost device is a server error");
  Check(service->device_lost(), "the service remembers the lost device");

  // A runner that returns fewer vectors than inputs is a server bug.
  EmbeddingService short_service(EmbeddingServiceOptions{
      .model_root = {},
      .validate_model = false,
      .model_id = "embeddinggemma-2",
      .prompts = {},
      .default_prompt_name = {},
      .runner =
          [](const std::vector<EmbeddingInput>&,
             const gufo::server::EmbeddingRequestOptions&,
             EmbeddingBatch* batch, std::string*) {
            batch->embeddings = {FakeEmbedding("only")};
            batch->token_counts = {3};
            return EmbeddingStatus::kOk;
          },
  });
  ExpectError(Send(short_service, R"({"input":["a","b"]})"), 500,
              "embedding_failed", "inconsistent batch");
}

void TestTypedInput() {
  std::vector<EmbeddingInput> captured;
  gufo::server::EmbeddingRequestOptions captured_options;
  EmbeddingService service(EmbeddingServiceOptions{
      .model_root = {},
      .validate_model = false,
      .model_id = "embeddinggemma-2",
      .prompts = {},
      .default_prompt_name = {},
      .runner = [&](const std::vector<EmbeddingInput>& inputs,
                    const gufo::server::EmbeddingRequestOptions& options,
                    EmbeddingBatch* batch, std::string*) {
        captured = inputs;
        captured_options = options;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
          batch->embeddings.push_back(FakeEmbedding(std::to_string(i)));
          batch->token_counts.push_back(i + 2);
        }
        return EmbeddingStatus::kOk;
      }});
  const auto response = Send(
      service,
      R"({"input":["plain",{"content":[{"type":"text","text":"before"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},{"type":"text","text":"after"}]},{"content":[{"type":"text","text":"typed"}]}]})");
  Check(response.status == 200, "typed input succeeds");
  Check(captured.size() == 3 && captured[1].content.size() == 3,
        "typed inputs preserve item and part counts");
  Check(captured[0].content[0].value == "plain" &&
            captured[1].content[0].value == "before" &&
            captured[1].content[1].type == EmbeddingPart::Type::kImage &&
            captured[1].content[1].value == "data:image/png;base64,AAAA" &&
            captured[1].content[2].value == "after" &&
            captured[2].content[0].value == "typed",
        "typed inputs preserve order and image URL");
  for (const auto* body :
       {R"({"input":[{"content":[]}]})",
        R"({"input":[{"content":[{"type":"text","text":""}]}]})"})
    ExpectError(Send(service, body), 400, "empty_input", "empty typed input");
  for (
      const auto* body :
      {R"({"input":[{"content":"x"}]})", R"({"input":[{"content":[true]}]})",
       R"({"input":[{"content":[{"type":"text","text":3}]}]})",
       R"({"input":[{"content":[{"type":"image_url","image_url":"x"}]}]})",
       R"({"input":[{"content":[{"type":"image_url","image_url":{"url":""}}]}]})",
       R"({"input":[{"content":[{"type":"image_url","image_url":{"url":"x","detail":"low"}}]}]})"})
    ExpectError(Send(service, body), 400, "invalid_parameter_type",
                "malformed typed input");
  ExpectError(
      Send(
          service,
          R"({"input":[{"content":[{"type":"input_audio","input_audio":{}}]}]})"),
      400, "invalid_parameter_type", "audio source is required");
  const auto audio_response = Send(
      service,
      R"({"input":["plain",{"content":[{"type":"input_audio","input_audio":{"data":"AAAA","url":"ignored","format":"ignored"}},{"type":"text","text":"middle"},{"type":"input_audio","input_audio":{"url":"https://example.com/test.wav"}},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]})");
  Check(audio_response.status == 200 && captured.size() == 2 &&
            captured[1].content.size() == 4,
        "audio and mixed content are accepted");
  Check(captured[1].content[0].type == EmbeddingPart::Type::kAudio &&
            captured[1].content[0].value == "AAAA" &&
            captured[1].content[1].value == "middle" &&
            captured[1].content[2].type == EmbeddingPart::Type::kAudio &&
            captured[1].content[2].value == "https://example.com/test.wav" &&
            captured[1].content[3].type == EmbeddingPart::Type::kImage,
        "audio data precedence and part order match the request");
  for (
      const auto* body :
      {R"({"input":[{"content":[{"type":"input_audio","input_audio":null}]}]})",
       R"({"input":[{"content":[{"type":"input_audio","input_audio":{"data":7}}]}]})",
       R"({"input":[{"content":[{"type":"input_audio","input_audio":{"data":"AAAA","format":7}}]}]})",
       R"({"input":[{"content":[{"type":"input_audio","input_audio":{"data":"AAAA","other":"x"}}]}]})"})
    ExpectError(Send(service, body), 400, "invalid_parameter_type",
                "malformed audio part");
  std::string audio_many = R"({"input":[{"content":[)";
  for (unsigned i = 0; i < 33; ++i) {
    if (i)
      audio_many += ',';
    audio_many += R"({"type":"input_audio","input_audio":{"data":"AAAA"}})";
  }
  audio_many += "]}]}";
  ExpectError(Send(service, audio_many), 400, "too_many_audio",
              "audio count bounded before decoding");
  std::string many = R"({"input":[{"content":[)";
  for (unsigned i = 0; i < 33; ++i) {
    if (i)
      many += ',';
    many +=
        R"({"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}})";
  }
  many += "]}]}";
  ExpectError(Send(service, many), 400, "too_many_images",
              "image count bounded before reading");
  Check(
      Send(
          service,
          R"({"input":[{"content":[{"type":"text","text":"before"},{"type":"input_video","input_video":{"data":"AAAA","url":"ignored"}},{"type":"text","text":"after"}]}],"video_options":{"fps":0.5,"max_frames":3,"add_timestamps":true,"overflow_strategy":"truncate"}})")
              .status == 200,
      "video content and controls accepted");
  Check(captured[0].content.size() == 3 &&
            captured[0].content[1].type == EmbeddingPart::Type::kVideo &&
            captured[0].content[1].value == "AAAA" &&
            captured[0].content[0].value == "before" &&
            captured[0].content[2].value == "after",
        "video data precedence and interleaving");
  Check(captured_options.video.fps == 0.5 &&
            captured_options.video.max_frames == 3 &&
            captured_options.video.timestamps &&
            captured_options.video.truncate_overflow,
        "video controls reach runtime");
  for (
      const auto* body :
      {R"({"input":[{"content":[{"type":"input_video","input_video":{}}]}]})",
       R"({"input":[{"content":[{"type":"input_video","input_video":{"data":3}}]}]})",
       R"({"input":[{"content":[{"type":"input_video","input_video":{"url":"x","extra":"x"}}]}]})"})
    ExpectError(Send(service, body), 400, "invalid_parameter_type",
                "invalid video schema");
  for (const auto* settings :
       {R"({"fps":0})", R"({"fps":121})", R"({"max_frames":0})",
        R"({"max_frames":33})", R"({"max_frames":1.5})",
        R"({"add_timestamps":1})", R"({"overflow_strategy":"other"})",
        R"({"unknown":true})"})
    ExpectError(
        Send(service, std::string(R"({"input":"hello","video_options":)") +
                          settings + "}"),
        400, "invalid_video_options", "invalid video control");
  std::string video_many = R"({"input":[{"content":[)";
  for (int i = 0; i < 33; ++i) {
    if (i)
      video_many += ',';
    video_many += R"({"type":"input_video","input_video":{"data":"AAAA"}})";
  }
  video_many += "]}]}";
  ExpectError(Send(service, video_many), 400, "too_many_videos",
              "video request count bound");
}

// #764: llama.cpp's non-OpenAI `POST /embeddings`, as Frigate 0.18 sends it
// (frigate/genai/plugins/llama_cpp.py at v0.18.2): `content` items with a
// `prompt_string` and base64 `multimodal_data`, and a bare-array response.
void TestNativeEmbeddings() {
  Check(IsEmbeddingsApiPath("/embeddings") &&
            IsEmbeddingsApiPath("/embedding") &&
            !IsEmbeddingsApiPath("/embeddings/") &&
            !IsEmbeddingsApiPath("/v1/embedding"),
        "native embeddings paths are recognized exactly");
  std::vector<EmbeddingInput> captured;
  int calls = 0;
  EmbeddingService service(EmbeddingServiceOptions{
      .model_root = {},
      .validate_model = false,
      .model_id = "embeddinggemma-2",
      .prompts = {},
      .default_prompt_name = {},
      .runner = [&](const std::vector<EmbeddingInput>& inputs,
                    const gufo::server::EmbeddingRequestOptions&,
                    EmbeddingBatch* batch, std::string*) {
        ++calls;
        captured = inputs;
        for (const auto& input : inputs) {
          std::string text;
          for (const auto& part : input.content)
            text += part.value;
          batch->embeddings.push_back(FakeEmbedding(text));
          batch->token_counts.push_back(FakeTokens(text));
        }
        return EmbeddingStatus::kOk;
      }});

  // Frigate's text request.
  const auto text = SendTo(
      service, "/embeddings",
      R"({"model":"embeddinggemma-2","content":[{"prompt_string":"a red car"},{"prompt_string":"a dog"}]})");
  Check(text.status == 200, "native text request succeeds: " + text.body);
  const gufo::json::Value body = Parse(text);
  Check(body.is_array() && body.size() == 2,
        "native response is a bare array with one item per input");
  for (std::size_t i = 0; i < 2; ++i) {
    const auto& item = body.items()[i];
    Check(item.size() == 2 && item.member_size("index", 99) == i,
          "native items carry only index and embedding, in order");
    const auto* nested = item.find("embedding");
    Check(nested != nullptr && nested->is_array() && nested->size() == 1,
          "the pooled vector is nested in a one-element list");
    Check(FloatsOf(nested->items().front()) ==
              FakeEmbedding(i == 0 ? "a red car" : "a dog"),
          "native floats round-trip exactly");
  }
  Check(captured.size() == 2 && captured[0].content.size() == 1 &&
            captured[0].content[0].type == EmbeddingPart::Type::kText &&
            captured[0].content[0].value == "a red car",
        "a prompt_string without media is one text part");

  // Frigate's image request: the marker, a newline, and one base64 JPEG.
  const auto image = SendTo(
      service, "/embeddings",
      R"({"model":"embeddinggemma-2","content":[{"prompt_string":"<__media__>\n","multimodal_data":["/9j/AAAA"]}]})");
  Check(image.status == 200, "native image request succeeds: " + image.body);
  Check(captured.size() == 1 && captured[0].content.size() == 2 &&
            captured[0].content[0].type == EmbeddingPart::Type::kImage &&
            captured[0].content[0].value == "data:image/jpeg;base64,/9j/AAAA" &&
            captured[0].content[1].type == EmbeddingPart::Type::kText &&
            captured[0].content[1].value == "\n",
        "the marker becomes the image part, in place, before the newline");
  const auto around = SendTo(
      service, "/embedding",
      R"({"content":{"prompt_string":"before <__media__> mid <__media__>","multimodal_data":["AAAA","BBBB"]}})");
  Check(around.status == 200, "legacy path and a single prompt object work");
  Check(captured.size() == 1 && captured[0].content.size() == 4 &&
            captured[0].content[0].value == "before " &&
            captured[0].content[1].value == "data:image/jpeg;base64,AAAA" &&
            captured[0].content[2].value == " mid " &&
            captured[0].content[3].value == "data:image/jpeg;base64,BBBB",
        "text around markers keeps its order and whitespace");

  // Like llama.cpp's mtmd helper, files that look like WAV, MP3 or FLAC
  // (and Ogg, which the audio decoder reads) become audio parts; WebP's RIFF
  // container and PNG stay images.
  const auto media = SendTo(
      service, "/embeddings",
      R"({"content":{"prompt_string":"<__media__><__media__><__media__><__media__><__media__><__media__><__media__>","multimodal_data":["UklGRiQAAABXQVZFZm10IA==","SUQzBAAAAAAAAAAA","//uQAAAAAAAAAAAA","ZkxhQwAAACIAAAAA","T2dnUwACAAAAAAAA","UklGRiQAAABXRUJQVlA4IA==","iVBORw0KGgoAAAANSUhEUg=="]}})");
  Check(media.status == 200, "native audio files succeed: " + media.body);
  using Type = EmbeddingPart::Type;
  const std::vector<Type> media_types{Type::kAudio, Type::kAudio, Type::kAudio,
                                      Type::kAudio, Type::kAudio, Type::kImage,
                                      Type::kImage};
  bool sniffed =
      captured.size() == 1 && captured[0].content.size() == media_types.size();
  for (std::size_t i = 0; sniffed && i < media_types.size(); ++i)
    sniffed = captured[0].content[i].type == media_types[i];
  Check(sniffed && captured[0].content[0].value == "UklGRiQAAABXQVZFZm10IA==",
        "multimodal_data audio is sniffed by its bytes and kept as raw base64");

  // The same text embeds identically on both routes; `input` wins over
  // `content`, as in llama.cpp.
  const auto plain =
      SendTo(service, "/embeddings", R"({"input":"x","content":"y"})");
  Check(plain.status == 200, "input and content together succeed");
  Check(captured.size() == 1 && captured[0].content[0].value == "x",
        "input takes precedence over content");
  const auto oai = Send(service, R"({"input":"x"})");
  const gufo::json::Value native_body = Parse(plain);
  const gufo::json::Value oai_body = Parse(oai);
  Check(FloatsOf(native_body.items().front().find("embedding")->items()[0]) ==
            FloatsOf(*oai_body.find("data")->items().front().find("embedding")),
        "both routes return the same vector");
  Check(
      SendTo(service, "/embeddings",
             R"({"content":"y","embd_normalize":2,"encoding_format":"float"})")
              .status == 200,
      "the default normalization and float encoding are accepted");
  const auto base64 = SendTo(service, "/embeddings",
                             R"({"content":"y","encoding_format":"base64"})");
  Check(base64.status == 200 && Parse(base64)
                                    .items()
                                    .front()
                                    .find("embedding")
                                    ->items()[0]
                                    .is_array(),
        "as in llama.cpp, the native form returns floats for base64 too");
  Check(SendTo(service, "/embeddings",
               R"({"content":[{"content":[{"type":"text","text":"typed"}]}]})")
                .status == 200,
        "typed content objects work here too");

  const int before = calls;
  ExpectError(SendTo(service, "/embeddings", R"({"model":"embeddinggemma-2"})"),
              400, "missing_input", "native request without input or content");
  ExpectError(
      SendTo(
          service, "/embeddings",
          R"({"content":[{"prompt_string":"<__media__><__media__>","multimodal_data":["AAAA"]}]})"),
      400, "invalid_multimodal_data", "more markers than files");
  ExpectError(
      SendTo(
          service, "/embeddings",
          R"({"content":[{"prompt_string":"x","multimodal_data":["AAAA"]}]})"),
      400, "invalid_multimodal_data", "a file without a marker");
  for (
      const auto* bad :
      {R"({"content":[{"prompt_string":3}]})",
       R"({"content":[{"prompt_string":"x","extra":1}]})",
       R"({"content":[{"prompt_string":"x","multimodal_data":"AAAA"}]})",
       R"({"content":[{"prompt_string":"<__media__>","multimodal_data":[""]}]})",
       R"({"content":[{"prompt_string":"<__media__>","multimodal_data":[7]}]})"})
    ExpectError(SendTo(service, "/embeddings", bad), 400,
                "invalid_parameter_type",
                std::string("malformed prompt: ") + bad);
  ExpectError(
      SendTo(service, "/embeddings", R"({"content":[{"prompt_string":""}]})"),
      400, "empty_input", "empty prompt_string");
  ExpectError(SendTo(service, "/embeddings", R"({"content":[1,2,3]})"), 400,
              "unsupported_input", "token arrays stay unsupported");
  for (const auto* normalize : {"-1", "0", "1", "3", "\"2\""})
    ExpectError(SendTo(service, "/embeddings",
                       std::string(R"({"content":"y","embd_normalize":)") +
                           normalize + "}"),
                400, "unsupported_embd_normalize",
                std::string("embd_normalize ") + normalize);
  ExpectError(SendTo(service, "/embeddings",
                     R"({"content":"y","encoding_format":"int8"})"),
              400, "invalid_encoding_format", "encoding_format is validated");
  ExpectError(
      SendTo(service, "/embeddings", R"({"content":"y","pooling":"none"})"),
      400, "unsupported_field", "unknown native fields are rejected");
  Check(calls == before, "rejected native requests never reach the runner");
  // The OpenAI route keeps its own field set.
  ExpectError(Send(service, R"({"content":"y"})"), 400, "unsupported_field",
              "/v1/embeddings does not take content");
  ExpectError(Send(service, R"({"input":"y","embd_normalize":2})"), 400,
              "unsupported_field",
              "/v1/embeddings does not take embd_normalize");
  ExpectError(Send(service, R"({"input":[{"prompt_string":"y"}]})"), 400,
              "invalid_parameter_type",
              "/v1/embeddings does not take prompt objects");
}

/// A response's structure: array lengths, member names and float-vector
/// widths, in the fixture's form.
gufo::json::Value Skeleton(const gufo::json::Value& value) {
  if (value.is_array()) {
    if (!value.items().empty() &&
        std::ranges::all_of(
            value.items(), [](const auto& item) { return item.is_number(); })) {
      auto floats = gufo::json::Value::object();
      floats["floats"] = value.size();
      return floats;
    }
    auto list = gufo::json::Value::object();
    list["list"] = value.size();
    if (!value.items().empty())
      list["item"] = Skeleton(value.items().front());
    return list;
  }
  if (value.is_object()) {
    auto object = gufo::json::Value::object();
    for (const auto& [name, member] : value.members())
      object[name] = Skeleton(member);
    return object;
  }
  return value;
}

// #764: replay llama.cpp's captured responses and Frigate's requests.
void TestNativeReferenceFixture() {
  std::ifstream input(GUFO_EMBEDDING_GEMMA2_NATIVE_FIXTURE);
  std::stringstream text;
  text << input.rdbuf();
  const gufo::json::Value fixture = gufo::json::parse(text.str());
  FakeRunner runner;
  auto service = MakeService(&runner);
  const auto& cases = *fixture.find("llama_cpp")->find("cases");
  Check(cases.size() >= 6, "fixture holds the llama.cpp cases");
  for (const auto& item : cases.items()) {
    gufo::json::Value request = *item.find("request");
    // The capture served another model name.
    if (request.contains("model"))
      request["model"] = "embeddinggemma-2";
    const auto response =
        SendTo(*service, item.member_str("path"), request.dump());
    Check(response.status == static_cast<int>(item.member_size("status")),
          "llama.cpp case status: " + request.dump() + " -> " + response.body);
    Check(Skeleton(Parse(response)).dump() == item.find("response")->dump(),
          "llama.cpp response shape for " + request.dump() + ": " +
              Skeleton(Parse(response)).dump());
  }
  const auto& frigate = *fixture.find("frigate");
  const auto path = frigate.member_str("path");
  const auto texts = SendTo(*service, path, frigate.find("text")->dump());
  Check(texts.status == 200 && Parse(texts).size() == 2,
        "Frigate's text request returns one vector per prompt");
  const auto image = SendTo(*service, path, frigate.find("image")->dump());
  Check(image.status == 200 && Parse(image).size() == 1,
        "Frigate's image request returns one vector");
}

// #726: `truncate: true` asks the service to cut an over-length input
// instead of refusing it. The default and `false` keep refusing.
void TestTruncate() {
  FakeRunner runner;
  auto service = MakeService(&runner);
  const HttpResponse refused =
      Send(*service, R"({"input":["short","TOOLONG input"]})");
  ExpectError(refused, 400, "context_length_exceeded", "default refuses");
  Check(!runner.last_truncate, "truncation is off by default");

  ExpectError(Send(*service, R"({"input":"TOOLONG input","truncate":false})"),
              400, "context_length_exceeded", "truncate false refuses");
  Check(!runner.last_truncate, "truncate false reaches the service");

  const HttpResponse cut =
      Send(*service, R"({"input":["short","TOOLONG input"],"truncate":true})");
  Check(cut.status == 200, "truncate true embeds an over-length input");
  Check(runner.last_truncate, "truncate true reaches the service");
  const auto body = Parse(cut);
  Check(body.find("data")->items().size() == 2, "both inputs are embedded");
  Check(body.find("usage")->member_size("prompt_tokens") ==
            FakeTokens("short") + EmbeddingService::kMaxInputTokens,
        "usage counts the tokens embedded after truncation");

  ExpectError(Send(*service, R"({"input":"a","truncate":"yes"})"), 400,
              "invalid_parameter_type", "truncate must be a boolean");
  const int calls = runner.calls;
  ExpectError(Send(*service, R"({"input":"a","truncate":1})"), 400,
              "invalid_parameter_type", "a numeric truncate is refused");
  Check(runner.calls == calls, "an invalid truncate never reaches the service");
  Check(Send(*service, R"({"input":"a","truncate":null})").status == 200,
        "a null truncate is absent");
}

// #763: `mm_processor_kwargs.max_soft_tokens` selects the image budget, as
// in vLLM's embedding requests; the default is the checkpoint's 280.
void TestImageSoftTokens() {
  FakeRunner runner;
  auto service = MakeService(&runner);
  Check(Send(*service, R"({"input":"a"})").status == 200, "default request");
  Check(runner.last_image_soft_tokens == 280, "the default budget is 280");
  for (const std::uint32_t budget : {70U, 140U, 280U, 560U, 1120U}) {
    const auto response = Send(
        *service, R"({"input":"a","mm_processor_kwargs":{"max_soft_tokens":)" +
                      std::to_string(budget) + "}}");
    Check(response.status == 200, "a supported budget is accepted");
    Check(runner.last_image_soft_tokens == budget,
          "the budget reaches the service");
  }
  Check(
      SendTo(*service, "/embeddings",
             R"({"content":"a","mm_processor_kwargs":{"max_soft_tokens":70}})")
              .status == 200,
      "the native form accepts the budget");
  Check(runner.last_image_soft_tokens == 70, "the native budget is applied");
  Check(Send(*service, R"({"input":"a","mm_processor_kwargs":null})").status ==
            200,
        "null mm_processor_kwargs is absent");
  Check(runner.last_image_soft_tokens == 280, "null keeps the default");
  Check(
      Send(*service, R"({"input":"a","mm_processor_kwargs":{}})").status == 200,
      "empty mm_processor_kwargs keeps the default");
  const int calls = runner.calls;
  ExpectError(Send(*service, R"({"input":"a","mm_processor_kwargs":[]})"), 400,
              "invalid_parameter_type", "mm_processor_kwargs is an object");
  for (const char* body :
       {R"({"input":"a","mm_processor_kwargs":{"max_soft_tokens":281}})",
        R"({"input":"a","mm_processor_kwargs":{"max_soft_tokens":0}})",
        R"({"input":"a","mm_processor_kwargs":{"max_soft_tokens":-70}})",
        R"({"input":"a","mm_processor_kwargs":{"max_soft_tokens":null}})",
        R"({"input":"a","mm_processor_kwargs":{"max_soft_tokens":true}})",
        R"({"input":"a","mm_processor_kwargs":{"max_soft_tokens":[]}})",
        R"({"input":"a","mm_processor_kwargs":{"max_soft_tokens":{}}})",
        R"({"input":"a","mm_processor_kwargs":{"max_soft_tokens":1121}})",
        R"({"input":"a","mm_processor_kwargs":{"max_soft_tokens":280.5}})",
        R"({"input":"a","mm_processor_kwargs":{"max_soft_tokens":"280"}})",
        R"({"input":"a","mm_processor_kwargs":{"max_soft_tokens":4294967576}})",
        R"({"input":"a","mm_processor_kwargs":{"fps":2}})"}) {
    ExpectError(Send(*service, body), 400, "invalid_mm_processor_kwargs",
                std::string("refused: ") + body);
  }
  Check(runner.calls == calls, "invalid kwargs never reach the service");
  Check(
      Send(
          *service,
          R"({"input":"a","mm_processor_kwargs":{"max_soft_tokens":70},"video_options":{"fps":0.5,"max_frames":3}})")
              .status == 200,
      "image and video processor options coexist");
  Check(runner.last_image_soft_tokens == 70,
        "video options do not override the image budget");
}

// The native path's cut for inputs with images: 2 = <bos>, 1 = <eos>,
// 9 = image placeholder, 8 = end-of-image marker, 5 = text.
void TestTruncateKeepingImages() {
  using Tokens = std::vector<std::uint32_t>;
  Tokens text{2, 5, 5, 5, 5, 5, 1};
  Check(TruncateKeepingImages(&text, 4, 9) && text == Tokens{2, 5, 5, 1},
        "a text-only encoding keeps its first tokens and <eos>");
  Tokens image{2, 5, 9, 9, 8, 5, 5, 5, 1};
  Check(
      TruncateKeepingImages(&image, 6, 9) && image == Tokens{2, 5, 9, 9, 8, 1},
      "a cut right after the end-of-image marker keeps the image");
  Tokens marker{2, 5, 9, 9, 8, 5, 5, 5, 1};
  Check(!TruncateKeepingImages(&marker, 5, 9) &&
            marker == Tokens{2, 5, 9, 9, 8, 5, 5, 5, 1},
        "a cut that would drop the end-of-image marker refuses unchanged");
  Tokens inside{2, 9, 9, 9, 8, 5, 1};
  Check(!TruncateKeepingImages(&inside, 3, 9), "a cut inside an image refuses");
  Tokens fits{2, 9, 8, 1};
  Check(TruncateKeepingImages(&fits, 4, 9) && fits == Tokens{2, 9, 8, 1},
        "an encoding within the limit is unchanged");
}

// #723: checkpoint prompts selected per request or by a server default.
std::unique_ptr<EmbeddingService> MakePromptService(FakeRunner* runner,
                                                    std::string default_name) {
  return std::make_unique<EmbeddingService>(EmbeddingServiceOptions{
      .model_root = {},
      .validate_model = false,
      .model_id = "embeddinggemma-2",
      .prompts = {{"query", "task: search result | query: "},
                  {"document", "title: none | text: "}},
      .default_prompt_name = std::move(default_name),
      .runner =
          [runner](const std::vector<EmbeddingInput>& inputs,
                   const EmbeddingRequestOptions& request,
                   EmbeddingBatch* batch, std::string* error) {
            std::vector<std::string> texts;
            for (const auto& input : inputs) {
              std::string text;
              for (const auto& part : input.content)
                text += part.value;
              texts.push_back(std::move(text));
            }
            return runner->Run(texts, request, batch, error);
          },
  });
}

void TestPromptNames() {
  FakeRunner runner;
  auto service = MakePromptService(&runner, "");
  Check(Send(*service, R"({"input":"a"})").status == 200, "no prompt");
  Check(runner.last_texts == std::vector<std::string>{"a"},
        "without a default, inputs reach the encoder unchanged");
  Check(Send(*service, R"({"input":["a","b"],"prompt_name":"query"})").status ==
            200,
        "a known prompt_name");
  Check(runner.last_texts ==
            std::vector<std::string>{"task: search result | query: a",
                                     "task: search result | query: b"},
        "prompt_name prefixes every input");
  const int calls = runner.calls;
  ExpectError(Send(*service, R"({"input":"a","prompt_name":"Query"})"), 400,
              "invalid_prompt_name", "prompt names are case-sensitive");
  ExpectError(Send(*service, R"({"input":"a","prompt_name":1})"), 400,
              "invalid_parameter_type", "a non-string prompt_name");
  Check(runner.calls == calls,
        "a refused prompt_name never reaches the service");

  auto with_default = MakePromptService(&runner, "document");
  Check(with_default->ready(), "a known default prompt");
  Check(Send(*with_default, R"({"input":"a"})").status == 200,
        "default prompt");
  Check(runner.last_texts == std::vector<std::string>{"title: none | text: a"},
        "the default prompt applies when the request names none");
  Check(Send(*with_default, R"({"input":"a","prompt_name":"query"})").status ==
                200 &&
            runner.last_texts ==
                std::vector<std::string>{"task: search result | query: a"},
        "a request's prompt_name overrides the default");
  Check(
      Send(*with_default, R"({"input":"a","prompt_name":""})").status == 200 &&
          runner.last_texts == std::vector<std::string>{"a"},
      "an empty prompt_name applies no prompt despite the default");
  Check(Send(*with_default, R"({"input":"a","prompt_name":null})").status ==
                200 &&
            runner.last_texts ==
                std::vector<std::string>{"title: none | text: a"},
        "a null prompt_name keeps the default");
  Check(SendTo(*with_default, "/embeddings",
               R"({"content":"a","prompt_name":"query"})")
                    .status == 200 &&
            runner.last_texts ==
                std::vector<std::string>{"task: search result | query: a"},
        "llama.cpp's /embeddings applies prompt_name too");

  // The model card passes images and audio without a prefix: the default
  // skips inputs without text, but a requested prompt applies to all.
  const std::string image =
      R"({"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}})";
  const std::string audio =
      R"({"type":"input_audio","input_audio":{"data":"AAAA"}})";
  const std::string caption = R"({"type":"text","text":"cap"})";
  const std::string media_inputs =
      R"({"input":[{"content":[)" + image + R"(]},{"content":[)" + audio +
      R"(]},{"content":[)" + image + "," + caption + R"(]},"a"]})";
  Check(Send(*with_default, media_inputs).status == 200 &&
            runner.last_texts.size() == 4,
        "default prompt with media inputs");
  Check(runner.last_texts[0] == "data:image/png;base64,AAAA",
        "the default prompt skips an image-only input");
  Check(!runner.last_texts[1].starts_with("title: none | text: "),
        "the default prompt skips an audio-only input");
  Check(runner.last_texts[2] ==
            "title: none | text: data:image/png;base64,AAAAcap",
        "the default prompt prefixes an input with text and an image");
  Check(runner.last_texts[3] == "title: none | text: a",
        "the default prompt prefixes text beside media inputs");
  Check(SendTo(*with_default, "/embeddings",
               R"({"content":{"prompt_string":"<__media__>",)"
               R"("multimodal_data":["AAAA"]}})")
                    .status == 200 &&
            !runner.last_texts[0].starts_with("title: none | text: "),
        "the default prompt skips llama.cpp media without text");
  Check(Send(*with_default, R"({"input":[{"content":[)" + image +
                                R"(]}],"prompt_name":"query"})")
                    .status == 200 &&
            runner.last_texts ==
                std::vector<std::string>{
                    "task: search result | query: data:image/png;base64,AAAA"},
        "a requested prompt_name prefixes an image-only input");

  auto unknown_default = MakePromptService(&runner, "missing");
  Check(!unknown_default->ready() &&
            unknown_default->initialization_error().find("missing") !=
                std::string::npos,
        "an unknown default prompt fails startup");

  const auto dir =
      std::filesystem::temp_directory_path() /
      ("embedding_prompts_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(dir);
  std::map<std::string, std::string> prompts{{"stale", "x"}};
  std::string error;
  Check(
      LoadSentenceTransformerPrompts(dir, &prompts, &error) && prompts.empty(),
      "a checkpoint without the file has no prompts");
  {
    std::ofstream out(dir / "config_sentence_transformers.json");
    out << R"({"default_prompt_name":null,"prompts":{"query":"q: ","document":"d: "}})";
  }
  Check(LoadSentenceTransformerPrompts(dir, &prompts, &error) &&
            prompts.size() == 2 && prompts.at("query") == "q: " &&
            prompts.at("document") == "d: ",
        "prompts load from config_sentence_transformers.json");
  {
    std::ofstream out(dir / "config_sentence_transformers.json");
    out << R"({"prompts":{"query":3}})";
  }
  Check(!LoadSentenceTransformerPrompts(dir, &prompts, &error) &&
            error.find("query") != std::string::npos,
        "a non-string prompt is refused");
  {
    std::ofstream out(dir / "config_sentence_transformers.json");
    out << R"({"prompts":["q: "]})";
  }
  Check(!LoadSentenceTransformerPrompts(dir, &prompts, &error) &&
            error.find("must be an object") != std::string::npos,
        "a non-object prompts table is refused");
  {
    std::ofstream out(dir / "config_sentence_transformers.json");
    out << R"({"prompts":)";
  }
  error.clear();
  Check(
      !LoadSentenceTransformerPrompts(dir, &prompts, &error) &&
          error.find("config_sentence_transformers.json") != std::string::npos,
      "malformed JSON is refused with the path");
  std::filesystem::remove_all(dir);
}

void TestCancellation() {
  FakeRunner runner;
  auto service = MakeService(&runner);
  ExpectError(Send(*service, R"({"input":"x"})", "POST", [] { return true; }),
              499, "cancelled", "cancelled before encoding");
  Check(runner.calls == 0, "a cancelled request never reaches the encoder");

  // Cancelled while the encoder runs. The client is gone, so no body is sent.
  std::atomic<bool> cancelled{false};
  EmbeddingService cancelling(EmbeddingServiceOptions{
      .model_root = {},
      .validate_model = false,
      .model_id = "embeddinggemma-2",
      .prompts = {},
      .default_prompt_name = {},
      .runner =
          [&cancelled](const std::vector<EmbeddingInput>& texts,
                       const gufo::server::EmbeddingRequestOptions&,
                       EmbeddingBatch* batch, std::string*) {
            cancelled = true;
            batch->embeddings = {FakeEmbedding(texts[0].content[0].value)};
            batch->token_counts = {3};
            return EmbeddingStatus::kOk;
          },
  });
  ExpectError(Send(cancelling, R"({"input":"x"})", "POST",
                   [&cancelled] { return cancelled.load(); }),
              499, "cancelled", "cancelled during encoding");
}

void TestServiceLifecycle() {
  // An injected runner needs no model files.
  FakeRunner runner;
  auto service = MakeService(&runner);
  Check(service->ready() && service->initialization_error().empty(),
        "fake service is ready");
  Check(service->model_id() == "embeddinggemma-2", "model id kept");
  EmbeddingService defaulted(EmbeddingServiceOptions{
      .model_root = {},
      .validate_model = false,
      .model_id = {},
      .prompts = {},
      .default_prompt_name = {},
      .runner = [](const std::vector<EmbeddingInput>&,
                   const gufo::server::EmbeddingRequestOptions&,
                   EmbeddingBatch*,
                   std::string*) { return EmbeddingStatus::kOk; },
  });
  Check(defaulted.model_id() == "embeddinggemma-2", "default model id");

  // Validation rejects a directory that is not an EmbeddingGemma 2 checkpoint.
  EmbeddingService invalid(EmbeddingServiceOptions{
      .model_root = "/nonexistent/embeddinggemma",
      .validate_model = true,
      .model_id = {},
      .prompts = {},
      .default_prompt_name = {},
      .runner = [](const std::vector<EmbeddingInput>&,
                   const gufo::server::EmbeddingRequestOptions&,
                   EmbeddingBatch*,
                   std::string*) { return EmbeddingStatus::kOk; },
  });
  Check(!invalid.ready() && !invalid.initialization_error().empty(),
        "an invalid model directory is refused");
  EmbeddingBatch batch;
  std::string error;
  Check(invalid.Embed({"x"}, {}, &batch, &error) == EmbeddingStatus::kFailed &&
            !error.empty(),
        "a service that is not ready refuses work");

#if !defined(ENGINE_ENABLE_HIP)
  EmbeddingService cpu_only(
      EmbeddingServiceOptions{.model_root = "/nonexistent",
                              .validate_model = false,
                              .model_id = {},
                              .prompts = {},
                              .default_prompt_name = {},
                              .runner = {}});
  Check(!cpu_only.ready() && cpu_only.initialization_error() ==
                                 "embedding service requires the HIP backend",
        "builds without HIP refuse to start");
#endif

  // GPU access is serialized.
  std::atomic<int> active{0};
  std::atomic<int> overlapped{0};
  EmbeddingService serial(EmbeddingServiceOptions{
      .model_root = {},
      .validate_model = false,
      .model_id = {},
      .prompts = {},
      .default_prompt_name = {},
      .runner =
          [&](const std::vector<EmbeddingInput>& texts,
              const gufo::server::EmbeddingRequestOptions&, EmbeddingBatch* out,
              std::string*) {
            if (active.fetch_add(1) != 0) {
              overlapped.fetch_add(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            out->embeddings = {FakeEmbedding(texts[0].content[0].value)};
            out->token_counts = {3};
            active.fetch_sub(1);
            return EmbeddingStatus::kOk;
          },
  });
  std::vector<std::jthread> threads;
  for (int i = 0; i < 6; ++i) {
    threads.emplace_back([&] {
      EmbeddingBatch result;
      Check(serial.Embed({"x"}, {}, &result) == EmbeddingStatus::kOk,
            "concurrent embed succeeds");
    });
  }
  threads.clear();
  Check(overlapped.load() == 0, "the encoder never runs concurrently");
}

}  // namespace

int main() {
  TestStringInput();
  TestArrayInputAndOrder();
  TestDimensions();
  TestBase64();
  TestInputValidation();
  TestModelAndJson();
  TestLimitsAndFailures();
  TestTypedInput();
  TestNativeEmbeddings();
  TestNativeReferenceFixture();
  TestTruncate();
  TestTruncateKeepingImages();
  TestImageSoftTokens();
  TestPromptNames();
  TestCancellation();
  TestServiceLifecycle();
  std::cout << "PASS embedding_gemma2_embeddings_api_test\n";
  return 0;
}
