// The rerank routes and service against an injected runner: no model files, no
// device and no socket. The test calls the HTTP handler directly.
#include "src/cli/serve/rerank_api.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gufo/ext/registry.hpp"
#include "gufo/ext/rerank.hpp"
#include "src/cli/serve/rerank_service.hpp"
#include "src/core/json.hpp"

namespace {

using gufo::ext::EmbeddingPart;
using gufo::server::HandleRerankApiRequest;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;
using gufo::server::IsRerankApiPath;
using gufo::server::RerankInput;
using gufo::server::RerankRequestOptions;
using gufo::server::RerankScores;
using gufo::server::RerankService;
using gufo::server::RerankServiceOptions;
using gufo::server::RerankStatus;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL rerank_api_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

// The score of a document is the number of its characters; the token count is
// that plus two (a template stand-in). Equal lengths tie.
float FakeScore(std::string_view document) {
  return static_cast<float>(document.size());
}

RerankInput TextInput(std::string text) {
  return {{{EmbeddingPart::Type::kText, std::move(text)}}};
}

// The text parts of an input, joined.
std::string Text(const RerankInput& input) {
  std::string text;
  for (const auto& part : input.content) {
    if (part.type == EmbeddingPart::Type::kText) {
      text += part.value;
    }
  }
  return text;
}

std::vector<RerankInput> TextInputs(std::vector<std::string> texts) {
  std::vector<RerankInput> inputs;
  for (auto& text : texts) {
    inputs.push_back(TextInput(std::move(text)));
  }
  return inputs;
}

struct Recorder {
  std::atomic<int> calls{0};
  // The text of the query and documents, and the parts as passed.
  std::string query;
  std::vector<std::string> documents;
  RerankInput query_input;
  std::vector<RerankInput> inputs;
  RerankRequestOptions options;
  RerankStatus status{RerankStatus::kOk};
  std::size_t failed_index{1};
  std::function<void(RerankScores*)> corrupt;
  std::function<void()> during;
};

struct Fixture {
  Recorder recorder;
  std::shared_ptr<RerankService> service;

  Fixture() {
    service = std::make_shared<RerankService>(RerankServiceOptions{
        .model_root = {},
        .validate_model = false,
        .model_id = "rerank-test",
        .model_aliases = {"rerank-alias"},
        .runner =
            [this](const RerankInput& query,
                   std::span<const RerankInput> documents,
                   const RerankRequestOptions& options,
                   const std::function<bool()>& is_cancelled,
                   RerankScores* scores, std::string* error) {
              ++recorder.calls;
              recorder.query = Text(query);
              recorder.query_input = query;
              recorder.documents.clear();
              for (const auto& document : documents) {
                recorder.documents.push_back(Text(document));
              }
              recorder.inputs.assign(documents.begin(), documents.end());
              recorder.options = options;
              if (recorder.during) {
                recorder.during();
              }
              if (is_cancelled && is_cancelled()) {
                return RerankStatus::kCancelled;
              }
              if (recorder.status != RerankStatus::kOk) {
                if (error) {
                  *error = "stub runner status";
                }
                scores->failed_index = recorder.failed_index;
                return recorder.status;
              }
              for (const auto& document : recorder.documents) {
                scores->scores.push_back(FakeScore(document));
                scores->token_counts.push_back(document.size() + 2);
              }
              if (recorder.corrupt) {
                recorder.corrupt(scores);
              }
              return RerankStatus::kOk;
            },
    });
  }

  HttpResponse Post(std::string_view path, std::string body,
                    std::function<bool()> cancelled = {}) {
    HttpRequest request;
    request.method = "POST";
    request.path = std::string(path);
    request.body = std::move(body);
    request.is_cancelled = std::move(cancelled);
    return HandleRerankApiRequest(request, *service);
  }
};

gufo::json::Value Parse(const HttpResponse& response) {
  try {
    return gufo::json::parse(response.body);
  } catch (const std::exception& error) {
    Fail(std::string("response is not JSON: ") + error.what() + ": " +
         response.body);
  }
}

std::string ErrorCode(const HttpResponse& response) {
  const auto root = Parse(response);
  const auto* error = root.find("error");
  Check(error != nullptr && error->is_object(), "error envelope");
  return error->member_str("code");
}

void ExpectError(const HttpResponse& response, int status,
                 const std::string& code, const std::string& what) {
  Check(response.status == status, what + ": status " +
                                       std::to_string(response.status) + " " +
                                       response.body);
  Check(ErrorCode(response) == code,
        what + ": code " + ErrorCode(response) + " expected " + code);
}

const char* kDocs = R"(["a","abcd","ab","abc"])";

void TestPaths() {
  Check(IsRerankApiPath("/v1/rerank"), "/v1/rerank");
  Check(IsRerankApiPath("/v1/reranking"), "/v1/reranking");
  Check(IsRerankApiPath("/rerank"), "/rerank");
  Check(!IsRerankApiPath("/v1/embeddings"), "other path");
  Check(!IsRerankApiPath("/v1/rerank/"), "trailing slash");
  Fixture fixture;
  HttpRequest get;
  get.method = "GET";
  get.path = "/v1/rerank";
  ExpectError(HandleRerankApiRequest(get, *fixture.service), 405,
              "method_not_allowed", "GET");
}

void TestJinaForm() {
  for (const char* path : {"/v1/rerank", "/v1/reranking", "/rerank"}) {
    Fixture fixture;
    const auto response = fixture.Post(
        path,
        std::string(R"({"model":"rerank-test","query":"q","documents":)") +
            kDocs + "}");
    Check(response.status == 200, std::string(path) + " status");
    const auto root = Parse(response);
    Check(root.member_str("model") == "rerank-test", "model");
    Check(root.member_str("object") == "list", "object");
    const auto* usage = root.find("usage");
    Check(usage != nullptr, "usage");
    // 3 + 6 + 4 + 5 tokens for the documents of length 1, 4, 2 and 3.
    Check(usage->member_size("prompt_tokens", 0) == 3 + 6 + 4 + 5,
          "prompt_tokens");
    Check(usage->member_size("total_tokens", 0) == 3 + 6 + 4 + 5,
          "total_tokens");
    const auto* results = root.find("results");
    Check(results != nullptr && results->is_array() && results->size() == 4,
          "results");
    const std::size_t expected[] = {1, 3, 2, 0};
    for (std::size_t i = 0; i < 4; ++i) {
      const auto& item = results->items()[i];
      Check(item.member_size("index", 99) == expected[i], "sorted index");
      Check(item.find("relevance_score") != nullptr, "relevance_score");
      Check(item.find("score") == nullptr, "no TEI score key");
      Check(item.find("text") == nullptr, "no text");
    }
    Check(results->items()[0].member_double("relevance_score", -1) == 4.0,
          "score value");
    Check(fixture.recorder.calls == 1, "one runner call for the request");
    Check(fixture.recorder.query == "q", "query passed once");
    Check(fixture.recorder.documents.size() == 4, "whole request passed");
    Check(response.headers.size() >= 1, "headers");
  }
}

void TestJinaWithoutModelNamesTheServedModel() {
  Fixture fixture;
  const auto response = fixture.Post(
      "/v1/rerank", std::string(R"({"query":"q","documents":)") + kDocs + "}");
  Check(response.status == 200, "no model member");
  Check(Parse(response).member_str("model") == "rerank-test", "served name");
}

void TestTopN() {
  Fixture fixture;
  const auto run = [&](const std::string& top_n) {
    return fixture.Post("/v1/rerank",
                        std::string(R"({"query":"q","documents":)") + kDocs +
                            R"(,"top_n":)" + top_n + "}");
  };
  {
    const auto root = Parse(run("2"));
    const auto* results = root.find("results");
    Check(results->size() == 2, "top_n 2");
    Check(results->items()[0].member_size("index", 99) == 1, "best first");
    Check(results->items()[1].member_size("index", 99) == 3, "second");
    // Usage counts every scored document, not just the returned ones.
    Check(root.find("usage")->member_size("prompt_tokens", 0) == 18, "usage");
  }
  Check(Parse(run("0")).find("results")->size() == 0, "top_n 0 is empty");
  Check(Parse(run("99")).find("results")->size() == 4, "top_n above count");
  ExpectError(run("-1"), 400, "invalid_top_n", "negative top_n");
  ExpectError(run("1.5"), 400, "invalid_parameter_type", "fractional top_n");
  ExpectError(run("\"2\""), 400, "invalid_parameter_type", "string top_n");
  const auto negative = run("-1");
  Check(negative.body.find("must be >= 0") != std::string::npos,
        "llama.cpp message");
}

void TestTiesKeepInputOrder() {
  Fixture fixture;
  const auto root = Parse(fixture.Post(
      "/v1/rerank", R"({"query":"q","documents":["bb","a","cc","dd","e"]})"));
  const auto* results = root.find("results");
  const std::size_t expected[] = {0, 2, 3, 1, 4};
  for (std::size_t i = 0; i < 5; ++i) {
    Check(results->items()[i].member_size("index", 99) == expected[i],
          "stable order");
  }
}

void TestTeiForm() {
  Fixture fixture;
  const auto response = fixture.Post(
      "/rerank", std::string(R"({"query":"q","texts":)") + kDocs + "}");
  Check(response.status == 200, "tei status");
  const auto root = Parse(response);
  Check(root.is_array() && root.size() == 4, "bare array");
  const std::size_t expected[] = {1, 3, 2, 0};
  for (std::size_t i = 0; i < 4; ++i) {
    const auto& item = root.items()[i];
    Check(item.member_size("index", 99) == expected[i], "tei order");
    Check(item.find("score") != nullptr, "score key");
    Check(item.find("relevance_score") == nullptr, "no Jina key");
    Check(item.find("text") == nullptr, "no text by default");
  }
  Check(response.body.find("usage") == std::string::npos, "no usage");

  const auto with_text = Parse(fixture.Post(
      "/rerank",
      std::string(R"({"query":"q","return_text":true,"top_n":2,"texts":)") +
          kDocs + "}"));
  Check(with_text.size() == 2, "tei top_n");
  Check(with_text.items()[0].member_str("text") == "abcd", "return_text");
  Check(with_text.items()[1].member_str("text") == "abc", "return_text 2");

  // `return_text` is read in the TEI form only.
  const auto jina = Parse(fixture.Post(
      "/rerank",
      std::string(R"({"query":"q","return_text":true,"documents":)") + kDocs +
          "}"));
  Check(jina.find("results")->items()[0].find("text") == nullptr,
        "return_text ignored in Jina form");
  ExpectError(
      fixture.Post("/rerank", R"({"query":"q","texts":["a"],"return_text":1})"),
      400, "invalid_parameter_type", "return_text type");
  // `texts` selects the TEI form even beside `documents`, which is the list
  // read when both are present, as in llama.cpp.
  const auto both = Parse(fixture.Post(
      "/rerank", R"({"query":"q","texts":["a","bb"],"documents":["ccc"]})"));
  Check(both.is_array(), "texts selects the TEI form");
}

void TestObjectForms() {
  Fixture fixture;
  const auto response = fixture.Post(
      "/v1/rerank",
      R"({"query":{"text":"find it"},"documents":[{"text":"abc"},"d"]})");
  Check(response.status == 200, "object forms");
  Check(fixture.recorder.query == "find it", "object query");
  Check(fixture.recorder.documents[0] == "abc", "object document");
}

void TestContentParts() {
  // The typed parts `/v1/embeddings` takes reach the runner in order.
  Fixture fixture;
  const auto response = fixture.Post(
      "/v1/rerank",
      R"({"query":{"content":[{"type":"text","text":"a red car"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]},)"
      R"("documents":["plain",{"text":"obj"},)"
      R"({"content":[{"type":"input_video","input_video":{"url":"https://example.com/v.mp4"}},{"type":"text","text":"clip"}]},)"
      R"({"content":[{"type":"input_video","input_video":{"data":"AAAA"}}]}]})");
  Check(response.status == 200, "content parts: " + response.body);
  const auto& query = fixture.recorder.query_input.content;
  Check(query.size() == 2 && query[0].type == EmbeddingPart::Type::kText &&
            query[0].value == "a red car" &&
            query[1].type == EmbeddingPart::Type::kImage &&
            query[1].value == "data:image/png;base64,AAAA",
        "query parts");
  const auto& inputs = fixture.recorder.inputs;
  Check(inputs.size() == 4, "four documents");
  Check(inputs[0].content.size() == 1 &&
            inputs[0].content[0].type == EmbeddingPart::Type::kText &&
            inputs[0].content[0].value == "plain",
        "a string is one text part");
  Check(inputs[1].content.size() == 1 && inputs[1].content[0].value == "obj",
        "a text object is one text part");
  Check(inputs[2].content.size() == 2 &&
            inputs[2].content[0].type == EmbeddingPart::Type::kVideo &&
            inputs[2].content[0].value == "https://example.com/v.mp4" &&
            inputs[2].content[1].type == EmbeddingPart::Type::kText,
        "video url then text, in order");
  Check(inputs[3].content.size() == 1 &&
            inputs[3].content[0].type == EmbeddingPart::Type::kVideo &&
            inputs[3].content[0].value == "AAAA",
        "video data");
  // TEI's `return_text` returns a document's text parts.
  const auto tei = Parse(fixture.Post(
      "/rerank",
      R"({"query":"q","texts":[{"content":[{"type":"text","text":"ab"},{"type":"image_url","image_url":{"url":"x"}},{"type":"text","text":"c"}]}],"return_text":true})"));
  Check(tei.is_array() && tei.items()[0].member_str("text") == "abc",
        "return_text joins text parts");
}

void TestContentPartErrors() {
  Fixture fixture;
  const auto document = [&](const std::string& item) {
    return fixture.Post("/v1/rerank",
                        R"({"query":"q","documents":[)" + item + "]}");
  };
  struct Case {
    const char* item;
    const char* code;
  };
  const Case cases[] = {
      {R"({"content":[]})", "invalid_parameter_type"},
      {R"({"content":"text"})", "invalid_parameter_type"},
      {R"({"content":[{"type":"text","text":"a"}],"text":"a"})",
       "invalid_parameter_type"},
      {R"({"content":["a"]})", "invalid_parameter_type"},
      {R"({"content":[{"type":"text"}]})", "invalid_parameter_type"},
      {R"({"content":[{"type":"text","text":"a","extra":1}]})",
       "invalid_parameter_type"},
      {R"({"content":[{"type":"image_url","image_url":{}}]})",
       "invalid_parameter_type"},
      {R"({"content":[{"type":"image_url","image_url":{"url":""}}]})",
       "invalid_parameter_type"},
      {R"({"content":[{"type":"image_url","image_url":"x"}]})",
       "invalid_parameter_type"},
      {R"({"content":[{"type":"input_video","input_video":{"data":""}}]})",
       "invalid_parameter_type"},
      {R"({"content":[{"type":"input_video","input_video":{"url":"x","fps":1}}]})",
       "invalid_parameter_type"},
      {R"({"content":[{"type":"input_audio","input_audio":{"data":"x"}}]})",
       "unsupported_input"},
      {R"({"content":[{"type":"other"}]})", "unsupported_input"},
      {R"({"content":[{"type":"text","text":""}]})", "empty_input"},
  };
  for (const auto& entry : cases) {
    ExpectError(document(entry.item), 400, entry.code, entry.item);
  }
  ExpectError(
      fixture.Post(
          "/v1/rerank",
          R"({"query":{"content":[{"type":"text","text":""}]},"documents":["a"]})"),
      400, "empty_input", "empty query content");
  Check(fixture.recorder.calls == 0,
        "a rejected request never reaches the runner");
}

void TestUnsupportedPartsAreRejected() {
  // A text-only runner turns an image part away; the client sees a 400.
  Fixture fixture;
  fixture.service = std::make_shared<RerankService>(RerankServiceOptions{
      .validate_model = false,
      .model_id = "text-only",
      .runner =
          [](const RerankInput& query, std::span<const RerankInput> documents,
             const RerankRequestOptions&, const std::function<bool()>&,
             RerankScores* scores, std::string* error) {
            const auto text_only = [](const RerankInput& input) {
              return std::ranges::all_of(input.content, [](const auto& part) {
                return part.type == EmbeddingPart::Type::kText;
              });
            };
            if (!text_only(query) ||
                !std::ranges::all_of(documents, text_only)) {
              *error = "text-only scores text only, not image parts";
              return RerankStatus::kUnsupportedInput;
            }
            scores->scores.assign(documents.size(), 1.0F);
            scores->token_counts.assign(documents.size(), 1);
            return RerankStatus::kOk;
          },
  });
  const auto response = fixture.Post(
      "/v1/rerank",
      R"({"query":"q","documents":["a",{"content":[{"type":"image_url","image_url":{"url":"x"}}]}]})");
  ExpectError(response, 400, "unsupported_input", "image for a text runner");
  Check(response.body.find("image parts") != std::string::npos,
        "the runner's message reaches the client: " + response.body);
  Check(
      fixture.Post("/v1/rerank", R"({"query":"q","documents":["a"]})").status ==
          200,
      "text still scores");
  const std::string text = fixture.service->RenderMetrics();
  Check(text.find("gufo_rerank_requests_total{status=\"invalid\"} 1") !=
            std::string::npos,
        "counted as invalid: " + text);
}

void TestNullMembersAreIgnored() {
  // As in llama.cpp, a null `image` is no image.
  Fixture fixture;
  const auto response = fixture.Post(
      "/v1/rerank",
      R"({"query":{"text":"q","image":null},"documents":[{"text":"a","image":null,"other":null}]})");
  Check(response.status == 200, "null members ignored: " + response.body);
  Check(fixture.recorder.query == "q", "query text kept");
}

void TestRequestErrors() {
  Fixture fixture;
  ExpectError(fixture.Post("/v1/rerank", "{"), 400, "parse_error", "bad JSON");
  ExpectError(fixture.Post("/v1/rerank", "[]"), 400, "invalid_request_body",
              "array body");
  ExpectError(fixture.Post("/v1/rerank", R"({"documents":["a"]})"), 400,
              "missing_query", "no query");
  ExpectError(fixture.Post("/v1/rerank", R"({"query":null,"documents":["a"]})"),
              400, "missing_query", "null query");
  ExpectError(fixture.Post("/v1/rerank", R"({"query":"","documents":["a"]})"),
              400, "empty_query", "empty query");
  ExpectError(fixture.Post("/v1/rerank", R"({"query":3,"documents":["a"]})"),
              400, "invalid_parameter_type", "numeric query");
  ExpectError(fixture.Post("/v1/rerank", R"({"query":"q"})"), 400,
              "invalid_documents", "no documents");
  ExpectError(fixture.Post("/v1/rerank", R"({"query":"q","documents":[]})"),
              400, "invalid_documents", "empty documents");
  ExpectError(fixture.Post("/v1/rerank", R"({"query":"q","texts":[]})"), 400,
              "invalid_documents", "empty texts");
  ExpectError(fixture.Post("/v1/rerank", R"({"query":"q","documents":"a"})"),
              400, "invalid_documents", "documents not an array");
  ExpectError(fixture.Post("/v1/rerank", R"({"query":"q","documents":[1]})"),
              400, "invalid_parameter_type", "numeric document");
  ExpectError(
      fixture.Post("/v1/rerank", R"({"query":"q","documents":["a",null]})"),
      400, "invalid_parameter_type", "null document");
  ExpectError(
      fixture.Post("/v1/rerank",
                   R"({"query":"q","documents":[{"text":"a","image":"x"}]})"),
      400, "unsupported_input", "image document");
  ExpectError(
      fixture.Post("/v1/rerank",
                   R"({"query":{"text":"q","image":"x"},"documents":["a"]})"),
      400, "unsupported_input", "image query");
  for (const char* member :
       {R"("video":"x")", R"("fps":1)", R"("max_frames":4)", R"("other":1)"}) {
    ExpectError(
        fixture.Post("/v1/rerank",
                     std::string(R"({"query":"q","documents":[{"text":"a",)") +
                         member + "}]}"),
        400, "unsupported_input", std::string("document ") + member);
    ExpectError(
        fixture.Post("/v1/rerank", std::string(R"({"query":{"text":"q",)") +
                                       member + R"(},"documents":["a"]})"),
        400, "unsupported_input", std::string("query ") + member);
  }
  ExpectError(
      fixture.Post("/v1/rerank",
                   R"({"query":"q","documents":["a"],"instruction":1})"),
      400, "invalid_parameter_type", "instruction type");
  ExpectError(
      fixture.Post("/v1/rerank",
                   R"({"query":"q","documents":["a"],"truncate":"yes"})"),
      400, "invalid_parameter_type", "truncate type");
  Check(fixture.recorder.calls == 0,
        "a rejected request never reaches the runner");
}

void TestDocumentCap() {
  Fixture fixture;
  Check(RerankService::kMaxDocuments > 40 * 10,
        "cap far above a 40-chunk search");
  const auto build = [](std::size_t count) {
    std::string body = R"({"query":"q","documents":[)";
    for (std::size_t i = 0; i < count; ++i) {
      body += i == 0 ? "\"x\"" : ",\"x\"";
    }
    return body + "]}";
  };
  const auto at_cap =
      fixture.Post("/v1/rerank", build(RerankService::kMaxDocuments));
  Check(at_cap.status == 200, "request at the cap");
  Check(Parse(at_cap).find("results")->size() == RerankService::kMaxDocuments,
        "all results");
  ExpectError(
      fixture.Post("/v1/rerank", build(RerankService::kMaxDocuments + 1)), 400,
      "too_many_documents", "above the cap");
}

void TestModelName() {
  Fixture fixture;
  const std::string tail = R"(,"query":"q","documents":["a"]})";
  ExpectError(fixture.Post("/v1/rerank", R"({"model":"other")" + tail), 404,
              "model_not_found", "other model");
  ExpectError(fixture.Post("/v1/rerank", R"({"model":3)" + tail), 400,
              "invalid_parameter_type", "model type");
  const auto aliased =
      fixture.Post("/v1/rerank", R"({"model":"rerank-alias")" + tail);
  Check(aliased.status == 200, "alias accepted");
  Check(Parse(aliased).member_str("model") == "rerank-test",
        "the primary name stays in the response");
  Check(fixture.service->serves_model("rerank-test"), "serves primary");
  Check(!fixture.service->serves_model("rerank"), "does not serve others");
}

void TestRequestOptionsReachTheRunner() {
  Fixture fixture;
  Check(
      fixture.Post(
                 "/v1/rerank",
                 R"({"query":"q","documents":["a"],"instruction":"Judge relevance","truncate":true})")
              .status == 200,
      "options request");
  Check(fixture.recorder.options.instruction == "Judge relevance",
        "instruction");
  Check(fixture.recorder.options.truncate, "truncate");
  Check(
      fixture.Post("/v1/rerank", R"({"query":"q","documents":["a"]})").status ==
          200,
      "defaults request");
  Check(fixture.recorder.options.instruction.empty(), "default instruction");
  Check(!fixture.recorder.options.truncate, "default truncate");
}

void TestStatusMapping() {
  Fixture fixture;
  const std::string body = R"({"query":"q","documents":["a","b"]})";
  struct Case {
    RerankStatus status;
    int http;
    const char* code;
  };
  const Case cases[] = {
      {RerankStatus::kCancelled, 499, "cancelled"},
      {RerankStatus::kInputTooLong, 400, "context_length_exceeded"},
      {RerankStatus::kInvalidInput, 400, "invalid_input"},
      {RerankStatus::kUnsupportedInput, 400, "unsupported_input"},
      {RerankStatus::kFailed, 500, "rerank_failed"},
  };
  for (const auto& entry : cases) {
    fixture.recorder.status = entry.status;
    ExpectError(fixture.Post("/v1/rerank", body), entry.http, entry.code,
                "status mapping");
  }
  fixture.recorder.status = RerankStatus::kInputTooLong;
  const auto too_long = fixture.Post("/v1/rerank", body);
  Check(too_long.body.find("document at index 1") != std::string::npos,
        "failed_index named");
  // An index outside the request is not repeated to the client.
  fixture.recorder.failed_index = 99;
  const auto stray = fixture.Post("/v1/rerank", body);
  ExpectError(stray, 400, "context_length_exceeded", "stray failed_index");
  Check(stray.body.find("index") == std::string::npos &&
            stray.body.find("99") == std::string::npos,
        "out-of-range index withheld: " + stray.body);
  fixture.recorder.status = RerankStatus::kFailed;
  Check(fixture.Post("/v1/rerank", body).body.find("stub runner status") !=
            std::string::npos,
        "runner message carried");
  Check(!fixture.service->device_lost(), "no device loss yet");
}

void TestDeviceLoss() {
  Fixture fixture;
  const std::string body = R"({"query":"q","documents":["a"]})";
  fixture.recorder.status = RerankStatus::kDeviceLost;
  ExpectError(fixture.Post("/v1/rerank", body), 503, "device_lost", "lost");
  Check(fixture.service->device_lost(), "the service stays lost");
  const int calls = fixture.recorder.calls;
  fixture.recorder.status = RerankStatus::kOk;
  ExpectError(fixture.Post("/v1/rerank", body), 503, "device_lost",
              "still lost");
  Check(fixture.recorder.calls == calls, "a lost device runs no more work");
}

void TestCancellation() {
  {
    // Cancelled before the runner starts: the runner is never called.
    Fixture fixture;
    ExpectError(fixture.Post("/v1/rerank", R"({"query":"q","documents":["a"]})",
                             [] { return true; }),
                499, "cancelled", "cancelled up front");
    Check(fixture.recorder.calls == 0, "runner not called");
  }
  {
    // The runner sees the cancellation the host forwards.
    Fixture fixture;
    std::atomic<bool> cancel{false};
    fixture.recorder.during = [&] { cancel = true; };
    ExpectError(fixture.Post("/v1/rerank", R"({"query":"q","documents":["a"]})",
                             [&] { return cancel.load(); }),
                499, "cancelled", "cancelled mid-run");
    Check(fixture.recorder.calls == 1, "runner called once");
  }
  {
    // A request cancelled after its scores arrive is still cancelled.
    Fixture fixture;
    std::atomic<bool> cancel{false};
    fixture.service = std::make_shared<RerankService>(RerankServiceOptions{
        .validate_model = false,
        .model_id = "rerank-test",
        .runner =
            [&](const RerankInput&, std::span<const RerankInput> documents,
                const RerankRequestOptions&, const std::function<bool()>&,
                RerankScores* scores, std::string*) {
              scores->scores.assign(documents.size(), 1.0F);
              scores->token_counts.assign(documents.size(), 3);
              cancel = true;
              return RerankStatus::kOk;
            },
    });
    ExpectError(fixture.Post("/v1/rerank", R"({"query":"q","documents":["a"]})",
                             [&] { return cancel.load(); }),
                499, "cancelled", "cancelled after run");
  }
}

void TestRunnerResultChecks() {
  const std::string body = R"({"query":"q","documents":["a","b"]})";
  {
    Fixture fixture;
    fixture.recorder.corrupt = [](RerankScores* scores) {
      scores->scores.pop_back();
    };
    ExpectError(fixture.Post("/v1/rerank", body), 500, "rerank_failed",
                "short scores");
  }
  {
    Fixture fixture;
    fixture.recorder.corrupt = [](RerankScores* scores) {
      scores->token_counts.clear();
    };
    ExpectError(fixture.Post("/v1/rerank", body), 500, "rerank_failed",
                "missing token counts");
  }
  {
    Fixture fixture;
    fixture.recorder.corrupt = [](RerankScores* scores) {
      scores->scores[1] = std::nanf("");
    };
    ExpectError(fixture.Post("/v1/rerank", body), 500, "rerank_failed",
                "NaN score");
  }
}

void TestMetrics() {
  Fixture fixture;
  Check(fixture.Post("/v1/rerank", R"({"query":"q","documents":["a","bb"]})")
                .status == 200,
        "metrics request");
  Check(fixture.Post("/v1/rerank", R"({"query":"q","documents":[]})").status ==
            400,
        "rejected before the service");
  fixture.recorder.status = RerankStatus::kInputTooLong;
  Check(
      fixture.Post("/v1/rerank", R"({"query":"q","documents":["a"]})").status ==
          400,
      "invalid request");
  const std::string text = fixture.service->RenderMetrics();
  const auto has = [&](const std::string& line) {
    return text.find(line) != std::string::npos;
  };
  Check(has("gufo_rerank_requests_total{status=\"ok\"} 1"),
        "ok count: " + text);
  Check(has("gufo_rerank_requests_total{status=\"invalid\"} 1"),
        "invalid count");
  Check(has("gufo_rerank_documents_total 3"), "documents count");
  Check(has("gufo_rerank_input_tokens_total 7"), "token count");
  Check(has("gufo_rerank_request_seconds_count 2"), "latency histogram");
  Check(has("gufo_rerank_requests_in_flight 0"), "in flight");
}

void TestScoreText() {
  // Scores are written as the shortest text that reads back to the same float.
  Fixture fixture;
  fixture.service = std::make_shared<RerankService>(RerankServiceOptions{
      .validate_model = false,
      .model_id = "m",
      .runner =
          [](const RerankInput&, std::span<const RerankInput> documents,
             const RerankRequestOptions&, const std::function<bool()>&,
             RerankScores* scores, std::string*) {
            scores->scores = {0.1F, -2.5F, 1.0F};
            scores->token_counts.assign(documents.size(), 1);
            return RerankStatus::kOk;
          },
  });
  const auto response =
      fixture.Post("/rerank", R"({"query":"q","texts":["a","b","c"]})");
  Check(
      response.body ==
          R"([{"index":2,"score":1},{"index":0,"score":0.1},{"index":1,"score":-2.5}])",
      "exact TEI body: " + response.body);
}

// A provider registered through the extension API serves requests, and the
// service passes it the model root and provider options.
class ProviderRunner final : public gufo::ext::RerankRunner {
public:
  gufo::ext::RerankStatus Rerank(
      const gufo::ext::RerankInput&,
      std::span<const gufo::ext::RerankInput> documents,
      const std::function<bool()>&, gufo::ext::RerankScores* out,
      std::string* error, const gufo::ext::RerankRequestOptions&) override {
    for (const auto& document : documents) {
      for (const auto& part : document.content) {
        if (part.type != EmbeddingPart::Type::kText) {
          if (error) {
            *error = "this reranker scores text only, not image parts";
          }
          return gufo::ext::RerankStatus::kUnsupportedInput;
        }
      }
      out->scores.push_back(static_cast<float>(Text(document).size()));
      out->token_counts.push_back(1);
    }
    return gufo::ext::RerankStatus::kOk;
  }
  std::string ModelId() const override { return "provider-model"; }
};

std::string g_provider_root;

bool ProviderProbe(std::string_view root) {
  return root.ends_with("rerank-provider-model");
}

bool ProviderResolve(std::string_view, std::string*) {
  return true;
}

std::unique_ptr<gufo::ext::RerankRunner> ProviderCreate(
    const gufo::ext::RerankRunnerOptions& options, std::string* error) {
  g_provider_root = options.model_root;
  if (options.model_root.ends_with("fail")) {
    *error = "cannot load";
    return nullptr;
  }
  return std::make_unique<ProviderRunner>();
}

void TestProviderRegistration() {
  gufo::ext::Registry registry;
  registry.AddRerankProvider({.id = "test-reranker",
                              .probe = ProviderProbe,
                              .resolve = ProviderResolve,
                              .create_with_options = ProviderCreate});
  Check(gufo::ext::FindRerankProvider("test-reranker") != nullptr, "find");
  Check(gufo::ext::FindRerankProvider("other") == nullptr, "find miss");
  Check(gufo::ext::ProbeRerankProvider("/models/rerank-provider-model") !=
            nullptr,
        "probe");
  Check(gufo::ext::ProbeRerankProvider("/models/elsewhere") == nullptr,
        "probe miss");
  // A second registration of the same id replaces the first.
  registry.AddRerankProvider({.id = "test-reranker",
                              .probe = ProviderProbe,
                              .resolve = ProviderResolve,
                              .create_with_options = ProviderCreate});

  RerankService service({.model_root = "/models/rerank-provider-model"});
  Check(service.ready(),
        "provider service ready: " + service.initialization_error());
  Check(g_provider_root == "/models/rerank-provider-model",
        "model root passed");
  Check(service.model_id() == "provider-model", "the runner names the model");
  Check(service.backend_name() == "extension", "backend name");
  RerankScores scores;
  Check(service.Rerank(TextInput("q"), TextInputs({"a", "bbb"}), {}, &scores) ==
            RerankStatus::kOk,
        "provider rerank");
  Check(scores.scores.size() == 2 && scores.scores[1] == 3.0F,
        "provider scores");
  RerankInput image = TextInput("caption");
  image.content.push_back({EmbeddingPart::Type::kImage, "x"});
  std::string error;
  Check(service.Rerank(TextInput("q"), std::vector<RerankInput>{image}, {},
                       &scores, &error) == RerankStatus::kUnsupportedInput &&
            error.find("image") != std::string::npos,
        "a text provider rejects an image part: " + error);

  RerankService named(
      {.model_root = "/models/rerank-provider-model", .model_id = "chosen"});
  Check(named.model_id() == "chosen", "served name wins");

  RerankService unknown({.model_root = "/models/elsewhere"});
  Check(!unknown.ready(), "no provider, not ready");
  Check(!unknown.initialization_error().empty(), "reason given");
  RerankScores none;
  Check(unknown.Rerank(TextInput("q"), TextInputs({"a"}), {}, &none) ==
            RerankStatus::kFailed,
        "a service that is not ready fails");
}

}  // namespace

int main() {
  TestPaths();
  TestJinaForm();
  TestJinaWithoutModelNamesTheServedModel();
  TestTopN();
  TestTiesKeepInputOrder();
  TestTeiForm();
  TestObjectForms();
  TestContentParts();
  TestContentPartErrors();
  TestUnsupportedPartsAreRejected();
  TestNullMembersAreIgnored();
  TestRequestErrors();
  TestDocumentCap();
  TestModelName();
  TestRequestOptionsReachTheRunner();
  TestStatusMapping();
  TestDeviceLoss();
  TestCancellation();
  TestRunnerResultChecks();
  TestMetrics();
  TestScoreText();
  TestProviderRegistration();
  std::cout << "rerank API checks passed.\n";
  return 0;
}
