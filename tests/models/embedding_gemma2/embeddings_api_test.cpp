// POST /v1/embeddings contract against a fake encoder: no model files, no
// device, and no socket. The test calls the HTTP handler directly.
#include "src/cli/serve/embeddings_api.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/embedding_service.hpp"
#include "src/core/json.hpp"

namespace {

using gufo::server::EmbeddingBatch;
using gufo::server::EmbeddingInput;
using gufo::server::EmbeddingPart;
using gufo::server::EmbeddingService;
using gufo::server::EmbeddingServiceOptions;
using gufo::server::EmbeddingStatus;
using gufo::server::HandleEmbeddingsApiRequest;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;
using gufo::server::IsEmbeddingsApiPath;

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
                      EmbeddingBatch* batch, std::string* error) {
    ++calls;
    last_texts = texts;
    for (std::size_t i = 0; i < texts.size(); ++i) {
      if (texts[i].starts_with("TOOLONG")) {
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
};

std::unique_ptr<EmbeddingService> MakeService(FakeRunner* runner) {
  return std::make_unique<EmbeddingService>(EmbeddingServiceOptions{
      .model_root = {},
      .validate_model = false,
      .model_id = "embeddinggemma-2",
      .runner =
          [runner](const std::vector<EmbeddingInput>& inputs,
                   EmbeddingBatch* batch, std::string* error) {
            std::vector<std::string> texts;
            for (const auto& input : inputs) {
              std::string text;
              for (const auto& part : input.content)
                text += part.value;
              texts.push_back(std::move(text));
            }
            return runner->Run(texts, batch, error);
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
  ExpectError(Send(*service, R"({"input":"x","truncate":true})"), 400,
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
      .runner =
          [](const std::vector<EmbeddingInput>&, EmbeddingBatch* batch,
             std::string*) {
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
  EmbeddingService service(EmbeddingServiceOptions{
      .model_root = {},
      .validate_model = false,
      .model_id = "embeddinggemma-2",
      .runner = [&](const std::vector<EmbeddingInput>& inputs,
                    EmbeddingBatch* batch, std::string*) {
        captured = inputs;
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
      400, "unsupported_input", "unsupported modality");
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
      .runner =
          [&cancelled](const std::vector<EmbeddingInput>& texts,
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
      .runner = [](const std::vector<EmbeddingInput>&, EmbeddingBatch*,
                   std::string*) { return EmbeddingStatus::kOk; },
  });
  Check(defaulted.model_id() == "embeddinggemma-2", "default model id");

  // Validation rejects a directory that is not an EmbeddingGemma 2 checkpoint.
  EmbeddingService invalid(EmbeddingServiceOptions{
      .model_root = "/nonexistent/embeddinggemma",
      .validate_model = true,
      .model_id = {},
      .runner = [](const std::vector<EmbeddingInput>&, EmbeddingBatch*,
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
      .runner =
          [&](const std::vector<EmbeddingInput>& texts, EmbeddingBatch* out,
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
  TestCancellation();
  TestServiceLifecycle();
  std::cout << "PASS embedding_gemma2_embeddings_api_test\n";
  return 0;
}
