#include "src/cli/eval/eval.hpp"

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/serve/json.hpp"
#include "src/eval/dataset.hpp"
#include "src/eval/extract.hpp"
#include "src/eval/http_client.hpp"

#ifndef GUFO_VERSION
#define GUFO_VERSION "development"
#endif
#ifndef GUFO_EVAL_SOURCE_DATA
#define GUFO_EVAL_SOURCE_DATA \
  "tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json"
#endif
#ifndef GUFO_EVAL_INSTALLED_DATA
#define GUFO_EVAL_INSTALLED_DATA GUFO_EVAL_SOURCE_DATA
#endif

namespace gufo::cli {
namespace {

using gufo::eval::HttpResult;
using gufo::server::json::Value;

constexpr std::size_t kMaximumQuestions = 75;
constexpr std::size_t kMaximumCompletionTokens = 16000;
constexpr std::string_view kDefaultBaseUrl = "http://127.0.0.1:8080/v1";

struct EvalOptions {
  std::string base_url{kDefaultBaseUrl};
  std::size_t questions{kMaximumQuestions};
  bool greedy{false};
  std::filesystem::path output;
};

struct ModelInfo {
  std::string id;
  std::string owned_by;
  std::string artifact_id;
  std::string revision;
};

struct SanitizedText {
  std::string text;
  std::size_t redactions{0};
};

struct ParsedCompletion {
  std::string content;
  std::string reasoning_content;
  std::string finish_reason;
  Value usage{Value::object()};
  Value timings{Value::object()};
  Value metrics{Value::object()};
};

std::filesystem::path DefaultEvalData() {
  std::filesystem::path source = GUFO_EVAL_SOURCE_DATA;
  std::error_code error;
  if (std::filesystem::is_regular_file(source, error)) {
    return source;
  }
  return GUFO_EVAL_INSTALLED_DATA;
}

std::string Environment(std::string_view name) {
  const char* value = std::getenv(std::string(name).c_str());
  return value == nullptr ? std::string() : std::string(value);
}

void ReplaceAll(std::string* text, std::string_view needle,
                std::string_view replacement, std::size_t* count) {
  if (text == nullptr || needle.empty()) {
    return;
  }
  std::size_t position = 0;
  while ((position = text->find(needle, position)) != std::string::npos) {
    text->replace(position, needle.size(), replacement);
    position += replacement.size();
    if (count != nullptr) {
      ++*count;
    }
  }
}

SanitizedText SanitizeText(std::string text, std::string_view base_url,
                           std::string_view bearer_token) {
  SanitizedText result{.text = std::move(text)};
  ReplaceAll(&result.text, base_url, "<endpoint>", &result.redactions);
  ReplaceAll(&result.text, bearer_token, "<credential>", &result.redactions);
  const std::string home = Environment("HOME");
  ReplaceAll(&result.text, home, "<home>", &result.redactions);

  const std::vector<std::pair<std::regex, std::string>> patterns = {
      {std::regex(R"(\b127(?:\.\d{1,3}){3}\b)"), "<private-address>"},
      {std::regex(R"(\b10(?:\.\d{1,3}){3}\b)"), "<private-address>"},
      {std::regex(R"(\b192\.168(?:\.\d{1,3}){2}\b)"), "<private-address>"},
      {std::regex(R"(\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b)"),
       "<private-address>"},
      {std::regex(R"(/home/[^/\s]+/)"), "/home/<user>/"},
      {std::regex(R"(/Users/[^/\s]+/)"), "/Users/<user>/"},
  };
  for (const auto& [pattern, replacement] : patterns) {
    const std::string updated =
        std::regex_replace(result.text, pattern, replacement);
    if (updated != result.text) {
      ++result.redactions;
      result.text = updated;
    }
  }
  return result;
}

Value CopyNumericTelemetry(const Value* source) {
  Value result = Value::object();
  if (source == nullptr || !source->is_object()) {
    return result;
  }

  struct PendingObject {
    const Value* source;
    std::vector<std::string> path;
  };
  std::vector<PendingObject> pending;
  pending.push_back({.source = source, .path = {}});
  for (std::size_t index = 0; index < pending.size(); ++index) {
    const PendingObject current = std::move(pending[index]);
    for (const auto& [key, value] : current.source->members()) {
      Value* destination = &result;
      for (const auto& component : current.path) {
        destination = &(*destination)[component];
      }
      if (value.is_number()) {
        (*destination)[key] = value.as_double();
      } else if (value.is_bool()) {
        (*destination)[key] = value.as_bool();
      } else if (value.is_object() && current.path.size() < 8) {
        std::vector<std::string> path = current.path;
        path.push_back(key);
        pending.push_back({.source = &value, .path = std::move(path)});
      }
    }
  }
  return result;
}

std::string SafeIdentity(std::string value, std::string_view base_url,
                         std::string_view bearer_token) {
  if (value.size() > 512) {
    value.resize(512);
  }
  return SanitizeText(std::move(value), base_url, bearer_token).text;
}

std::optional<ModelInfo> ParseSingleModel(const HttpResult& response,
                                          std::string* error) {
  if (!response.transport_ok) {
    if (error != nullptr) {
      *error =
          "model discovery transport failed (" + response.transport_code + ")";
    }
    return std::nullopt;
  }
  if (response.status_code < 200 || response.status_code >= 300) {
    if (error != nullptr) {
      *error = "model discovery returned HTTP " +
               std::to_string(response.status_code);
    }
    return std::nullopt;
  }

  try {
    const Value root = gufo::server::json::parse(response.body);
    const Value* data = root.find("data");
    if (data == nullptr || !data->is_array()) {
      throw std::runtime_error("response has no data array");
    }
    if (data->size() != 1) {
      if (error != nullptr) {
        *error = "expected exactly one model from GET /v1/models, found " +
                 std::to_string(data->size()) + "; start a single-model server";
      }
      return std::nullopt;
    }
    const Value& model = data->items().front();
    const Value* id = model.find("id");
    if (!model.is_object() || id == nullptr || !id->is_string() ||
        id->str().empty()) {
      throw std::runtime_error("sole model has no string id");
    }

    ModelInfo info;
    info.id = id->str();
    info.owned_by = model.member_str("owned_by");
    info.revision = model.member_str("revision");
    if (info.revision.empty()) {
      info.revision = model.member_str("model_revision");
    }
    const Value* gufo = model.find("gufo");
    if (gufo != nullptr && gufo->is_object()) {
      info.artifact_id = gufo->member_str("artifact_id");
      if (info.revision.empty()) {
        info.revision = gufo->member_str("repository_revision");
      }
    }
    return info;
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = "malformed model discovery response: " +
               std::string(exception.what());
    }
    return std::nullopt;
  }
}

Value MessagesJson(std::string_view system, std::string_view user) {
  Value messages = Value::array();
  Value system_message = Value::object();
  system_message["role"] = "system";
  system_message["content"] = std::string(system);
  messages.push_back(std::move(system_message));
  Value user_message = Value::object();
  user_message["role"] = "user";
  user_message["content"] = std::string(user);
  messages.push_back(std::move(user_message));
  return messages;
}

Value RequestJson(const ModelInfo& model, Value messages, bool greedy) {
  Value request = Value::object();
  request["model"] = model.id;
  request["messages"] = std::move(messages);
  request["max_completion_tokens"] = kMaximumCompletionTokens;
  request["stream"] = false;
  if (greedy) {
    request["temperature"] = 0;
  }
  return request;
}

std::optional<ParsedCompletion> ParseCompletion(const HttpResult& response,
                                                std::string* error) {
  try {
    const Value root = gufo::server::json::parse(response.body);
    const Value* choices = root.find("choices");
    if (choices == nullptr || !choices->is_array() || choices->size() != 1) {
      throw std::runtime_error("response must contain exactly one choice");
    }
    const Value& choice = choices->items().front();
    const Value* message = choice.find("message");
    const Value* finish_reason = choice.find("finish_reason");
    if (!choice.is_object() || message == nullptr || !message->is_object() ||
        finish_reason == nullptr || !finish_reason->is_string()) {
      throw std::runtime_error("choice has no message or finish_reason");
    }
    const Value* content = message->find("content");
    if (content == nullptr || !content->is_string()) {
      throw std::runtime_error("assistant message content is not a string");
    }

    ParsedCompletion parsed;
    parsed.content = content->str();
    parsed.finish_reason = finish_reason->str();
    const Value* reasoning = message->find("reasoning_content");
    if (reasoning != nullptr) {
      if (!reasoning->is_string()) {
        throw std::runtime_error("reasoning_content is not a string");
      }
      parsed.reasoning_content = reasoning->str();
    }
    parsed.usage = CopyNumericTelemetry(root.find("usage"));
    parsed.timings = CopyNumericTelemetry(root.find("timings"));
    parsed.metrics = CopyNumericTelemetry(root.find("metrics"));
    return parsed;
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = exception.what();
    }
    return std::nullopt;
  }
}

std::string ApiErrorCode(const HttpResult& response) {
  try {
    const Value root = gufo::server::json::parse(response.body);
    const Value* api_error = root.find("error");
    if (api_error != nullptr && api_error->is_object()) {
      return api_error->member_str("code");
    }
  } catch (const std::exception& exception) {
    (void)exception;
  }
  return {};
}

bool IsContextRejection(const HttpResult& response,
                        std::string_view error_code) {
  std::string combined = response.body + " " + std::string(error_code);
  std::ranges::transform(combined, combined.begin(), [](unsigned char value) {
    return static_cast<char>(std::tolower(value));
  });
  return combined.find("context") != std::string::npos &&
         (combined.find("length") != std::string::npos ||
          combined.find("window") != std::string::npos ||
          combined.find("token") != std::string::npos);
}

bool WriteReport(const std::filesystem::path& output, const Value& report,
                 std::string* error) {
  if (output.empty()) {
    if (error != nullptr) {
      *error = "--output is required";
    }
    return false;
  }
  std::filesystem::path temporary = output;
  temporary += ".tmp-" + std::to_string(static_cast<long long>(getpid()));
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
      if (error != nullptr) {
        *error = "cannot create output file";
      }
      return false;
    }
    stream << report.dump() << '\n';
    if (!stream) {
      if (error != nullptr) {
        *error = "failed while writing output file";
      }
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
      return false;
    }
  }
  std::error_code rename_error;
  std::filesystem::rename(temporary, output, rename_error);
  if (rename_error) {
    if (error != nullptr) {
      *error = "cannot finalize output file";
    }
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    return false;
  }
  return true;
}

std::string SanitizedCommand(const EvalOptions& options) {
  std::ostringstream command;
  command << "gufo eval";
  if (options.base_url != kDefaultBaseUrl) {
    command << " --base-url <endpoint>";
  }
  if (options.questions != kMaximumQuestions) {
    command << " --questions " << options.questions;
  }
  if (options.greedy) {
    command << " --greedy";
  }
  command << " --output <output>";
  return command.str();
}

Value OptionalIdentity(std::string value, std::string_view base_url,
                       std::string_view bearer_token) {
  if (value.empty()) {
    return Value();
  }
  return Value(SafeIdentity(std::move(value), base_url, bearer_token));
}

std::string EvaluatorState() {
  const std::string version = GUFO_VERSION;
  if (version == "development") {
    return "unknown";
  }
  return version.find("dirty") == std::string::npos ? "clean" : "dirty";
}

int Execute(const EvalOptions& options) {
  std::string error;
  if (!gufo::eval::ValidateBaseUrl(options.base_url, &error)) {
    std::cerr << "eval: " << error << '\n';
    return 2;
  }
  const auto suite = gufo::eval::LoadEvalSuite(DefaultEvalData(), &error);
  if (!suite.has_value()) {
    std::cerr << "eval: " << error << '\n';
    return 1;
  }
  if (options.questions == 0 || options.questions > suite->cases.size()) {
    std::cerr << "eval: --questions must be between 1 and "
              << suite->cases.size() << '\n';
    return 2;
  }
  if (options.output.empty()) {
    std::cerr << "eval: --output is required\n";
    return 2;
  }

  std::string bearer_token = Environment("OPENAI_API_KEY");
  if (bearer_token.empty()) {
    bearer_token = Environment("GUFO_EVAL_API_KEY");
  }
  const gufo::eval::HttpClient client(options.base_url, bearer_token);
  const auto model_response = client.Get("/models");
  const auto discovered = ParseSingleModel(model_response, &error);
  if (!discovered.has_value()) {
    std::cerr << "eval: " << error << '\n';
    return 1;
  }
  ModelInfo model = *discovered;
  const std::string model_revision = Environment("GUFO_EVAL_MODEL_REVISION");
  const std::string artifact_id = Environment("GUFO_EVAL_ARTIFACT_ID");
  if (!model_revision.empty()) {
    model.revision = model_revision;
  }
  if (!artifact_id.empty()) {
    model.artifact_id = artifact_id;
  }

  Value report = Value::object();
  report["schema"] = "gufo.eval-result.v1";
  report["benchmark"] = suite->benchmark;
  Value source = Value::object();
  source["repository"] = suite->source_repository;
  source["revision"] = suite->source_revision;
  source["path"] = suite->source_path;
  source["blob"] = suite->source_blob;
  report["benchmark_source"] = std::move(source);

  Value run = Value::object();
  run["evaluator_commit"] = GUFO_VERSION;
  run["evaluator_worktree"] = EvaluatorState();
  run["command"] = SanitizedCommand(options);
  run["questions"] = options.questions;
  run["request_order"] = "sequential";
  run["independent_requests"] = true;
  run["max_completion_tokens"] = kMaximumCompletionTokens;
  run["temperature"] = options.greedy ? Value(0) : Value();
  run["temperature_policy"] = options.greedy ? "greedy" : "server_default";
  run["thinking_policy"] = "server_default";
  Value server = Value::object();
  server["label"] = OptionalIdentity(Environment("GUFO_EVAL_SERVER_LABEL"),
                                     options.base_url, bearer_token);
  server["commit"] = OptionalIdentity(Environment("GUFO_EVAL_SERVER_COMMIT"),
                                      options.base_url, bearer_token);
  server["configuration"] = OptionalIdentity(
      Environment("GUFO_EVAL_SERVER_CONFIG"), options.base_url, bearer_token);
  run["server"] = std::move(server);
  Value model_json = Value::object();
  model_json["id"] = SafeIdentity(model.id, options.base_url, bearer_token);
  model_json["owned_by"] =
      OptionalIdentity(model.owned_by, options.base_url, bearer_token);
  model_json["artifact_id"] =
      OptionalIdentity(model.artifact_id, options.base_url, bearer_token);
  model_json["revision"] =
      OptionalIdentity(model.revision, options.base_url, bearer_token);
  run["model"] = std::move(model_json);
  report["run"] = std::move(run);

  Value cases = Value::array();
  std::size_t passed = 0;
  std::size_t failed = 0;
  std::size_t execution_errors = 0;
  std::size_t completion_tokens = 0;
  std::size_t length_finishes = 0;

  for (std::size_t index = 0; index < options.questions; ++index) {
    const auto& eval_case = suite->cases[index];
    const std::string user_prompt = gufo::eval::BuildUserPrompt(eval_case);
    Value request_messages = MessagesJson(suite->system_prompt, user_prompt);
    Value report_messages = MessagesJson(suite->system_prompt, user_prompt);
    const Value request =
        RequestJson(model, std::move(request_messages), options.greedy);
    const auto response = client.PostJson("/chat/completions", request.dump());

    Value item = Value::object();
    item["index"] = index;
    item["ds4_index"] = eval_case.ds4_index;
    item["source"] = eval_case.source;
    item["case_id"] = eval_case.id;
    item["kind"] = std::string(gufo::eval::AnswerKindName(eval_case.kind));
    item["messages"] = std::move(report_messages);
    item["expected_answer"] = eval_case.answer;

    Value execution = Value::object();
    execution["http_status"] =
        response.transport_ok ? Value(response.status_code) : Value();
    execution["transport_ms"] = response.elapsed_ms;
    Value endpoint_error;
    std::string status;
    std::optional<ParsedCompletion> completion;
    std::string parse_error;

    if (!response.transport_ok) {
      status = "transport_error";
      endpoint_error = response.transport_code;
      ++execution_errors;
    } else if (response.status_code < 200 || response.status_code >= 300) {
      const std::string error_code = ApiErrorCode(response);
      status = IsContextRejection(response, error_code) ? "context_rejected"
                                                        : "http_error";
      endpoint_error =
          error_code.empty()
              ? Value("http_" + std::to_string(response.status_code))
              : Value(SafeIdentity(error_code, options.base_url, bearer_token));
      ++execution_errors;
    } else {
      completion = ParseCompletion(response, &parse_error);
      if (!completion.has_value()) {
        status = "malformed_response";
        endpoint_error = "invalid_chat_completion";
        ++execution_errors;
      } else {
        status = completion->finish_reason;
        if (status == "length") {
          ++length_finishes;
        }
      }
    }
    execution["status"] = status;
    execution["error_code"] = std::move(endpoint_error);
    item["execution"] = std::move(execution);

    Value response_json = Value::object();
    Value grade = Value::object();
    if (completion.has_value()) {
      const auto content =
          SanitizeText(completion->content, options.base_url, bearer_token);
      const auto reasoning = SanitizeText(completion->reasoning_content,
                                          options.base_url, bearer_token);
      response_json["content"] = content.text;
      response_json["reasoning_content"] = completion->reasoning_content.empty()
                                               ? Value()
                                               : Value(reasoning.text);
      response_json["finish_reason"] = completion->finish_reason;
      response_json["redactions"] = content.redactions + reasoning.redactions;
      completion_tokens += completion->usage.member_size("completion_tokens");
      item["usage"] = std::move(completion->usage);
      item["timings"] = std::move(completion->timings);
      item["metrics"] = std::move(completion->metrics);

      const auto result =
          gufo::eval::GradeAnswer(eval_case, completion->content);
      grade["extraction_status"] = std::string(
          gufo::eval::ExtractionStatusName(result.extraction.status));
      grade["extracted_answer"] =
          result.extraction.status == gufo::eval::ExtractionStatus::kFound
              ? Value(result.extraction.answer)
              : Value();
      grade["verdict"] = result.passed ? "passed" : "failed";
      if (result.passed) {
        ++passed;
      } else {
        ++failed;
      }
    } else {
      response_json["content"] = Value();
      response_json["reasoning_content"] = Value();
      response_json["finish_reason"] = Value();
      response_json["redactions"] = 0;
      item["usage"] = Value::object();
      item["timings"] = Value::object();
      item["metrics"] = Value::object();
      grade["extraction_status"] = "not_run";
      grade["extracted_answer"] = Value();
      grade["verdict"] = "not_graded";
    }
    item["response"] = std::move(response_json);
    item["grade"] = std::move(grade);
    cases.push_back(std::move(item));

    std::cout << '[' << index + 1 << '/' << options.questions << "] "
              << eval_case.source << '/' << eval_case.id << ": ";
    if (!completion.has_value()) {
      std::cout << status;
    } else {
      std::cout << (cases.items().back().find("grade")->member_str("verdict",
                                                                   "failed"))
                << " (" << status << ')';
    }
    std::cout << '\n';
  }
  report["cases"] = std::move(cases);

  Value summary = Value::object();
  summary["total_cases"] = options.questions;
  summary["passed"] = passed;
  summary["failed"] = failed;
  summary["not_graded"] = execution_errors;
  summary["execution_errors"] = execution_errors;
  summary["length_finishes"] = length_finishes;
  summary["completion_tokens"] = completion_tokens;
  report["summary"] = std::move(summary);

  if (!WriteReport(options.output, report, &error)) {
    std::cerr << "eval: " << error << '\n';
    return 1;
  }
  std::cout << "eval: wrote " << options.questions << " cases to "
            << options.output << " (passed=" << passed << ", failed=" << failed
            << ", execution_errors=" << execution_errors << ")\n";
  return execution_errors == 0 ? 0 : 1;
}

EvalOptions ParseOptions(std::span<const char* const> args,
                         bool* help_requested, std::string* error) {
  EvalOptions options;
  ArgParser parser("gufo eval",
                   "Run the pinned Antirez DS4 capability questions against "
                   "an OpenAI-compatible server.");
  parser.AddOption("", "--base-url", "URL",
                   "OpenAI-compatible API root (default: "
                   "http://127.0.0.1:8080/v1)",
                   "Endpoint", &options.base_url);
  parser.AddOption("", "--questions", "N",
                   "Run the first N pinned cases (default: 75)", "Evaluation",
                   &options.questions);
  parser.AddFlag("", "--greedy",
                 "Send temperature: 0 (otherwise omit temperature)",
                 "Evaluation", &options.greedy);
  parser.AddOption("", "--output", "PATH",
                   "Write the sanitized machine-readable JSON result", "Output",
                   &options.output);
  if (!parser.Parse(args, error)) {
    return options;
  }
  if (help_requested != nullptr) {
    *help_requested = parser.IsHelpRequested();
  }
  return options;
}

}  // namespace

void PrintEvalHelp(std::string_view) {
  EvalOptions options;
  ArgParser parser("gufo eval",
                   "Run the pinned Antirez DS4 capability questions against "
                   "an OpenAI-compatible server.");
  parser.AddOption("", "--base-url", "URL",
                   "OpenAI-compatible API root (default: "
                   "http://127.0.0.1:8080/v1)",
                   "Endpoint", &options.base_url);
  parser.AddOption("", "--questions", "N",
                   "Run the first N pinned cases (default: 75)", "Evaluation",
                   &options.questions);
  parser.AddFlag("", "--greedy",
                 "Send temperature: 0 (otherwise omit temperature)",
                 "Evaluation", &options.greedy);
  parser.AddOption("", "--output", "PATH",
                   "Write the sanitized machine-readable JSON result", "Output",
                   &options.output);
  parser.PrintHelp();
}

int RunEval(std::span<const char* const> args) {
  bool help_requested = false;
  std::string error;
  const EvalOptions options = ParseOptions(args, &help_requested, &error);
  if (!error.empty()) {
    std::cerr << "eval: " << error << '\n';
    return 2;
  }
  if (help_requested) {
    PrintEvalHelp("gufo");
    return 0;
  }
  return Execute(options);
}

}  // namespace gufo::cli
