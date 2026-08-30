#include "src/eval/dataset.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include "src/cli/serve/json.hpp"

namespace gufo::eval {
namespace {

using gufo::server::json::Value;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

const Value& RequireMember(const Value& object, std::string_view key,
                           Value::Type type) {
  const Value* value = object.find(std::string(key));
  if (value == nullptr || value->type() != type) {
    throw std::runtime_error("missing or invalid '" + std::string(key) + "'");
  }
  return *value;
}

std::string RequireString(const Value& object, std::string_view key) {
  return RequireMember(object, key, Value::Type::kString).str();
}

AnswerKind ParseKind(std::string_view value) {
  if (value == "mcq") {
    return AnswerKind::kMultipleChoice;
  }
  if (value == "integer") {
    return AnswerKind::kInteger;
  }
  if (value == "linespec") {
    return AnswerKind::kLineSpec;
  }
  throw std::runtime_error("unknown answer kind '" + std::string(value) + "'");
}

EvalCase ParseCase(const Value& value) {
  if (!value.is_object()) {
    throw std::runtime_error("case is not an object");
  }
  EvalCase eval_case;
  eval_case.ds4_index =
      RequireMember(value, "ds4_index", Value::Type::kNumber).as_size();
  eval_case.source = RequireString(value, "source");
  eval_case.id = RequireString(value, "id");
  eval_case.domain = RequireString(value, "domain");
  eval_case.title = RequireString(value, "title");
  eval_case.kind = ParseKind(RequireString(value, "kind"));
  eval_case.question = RequireString(value, "question");
  eval_case.answer = RequireString(value, "answer");
  eval_case.dataset = RequireString(value, "dataset");
  eval_case.dataset_url = RequireString(value, "dataset_url");
  eval_case.license = RequireString(value, "license");
  eval_case.audit_note = RequireString(value, "audit_note");
  eval_case.source_record_sha256 = RequireString(value, "source_record_sha256");

  const Value& choices = RequireMember(value, "choices", Value::Type::kArray);
  eval_case.choices.reserve(choices.size());
  for (const auto& choice : choices.items()) {
    if (!choice.is_string()) {
      throw std::runtime_error("case choice is not a string");
    }
    eval_case.choices.push_back(choice.str());
  }
  if (eval_case.kind == AnswerKind::kMultipleChoice &&
      eval_case.choices.empty()) {
    throw std::runtime_error("multiple-choice case has no choices");
  }
  if (eval_case.kind != AnswerKind::kMultipleChoice &&
      !eval_case.choices.empty()) {
    throw std::runtime_error("non-MCQ case unexpectedly has choices");
  }
  return eval_case;
}

}  // namespace

std::optional<EvalSuite> LoadEvalSuite(const std::filesystem::path& path,
                                       std::string* error) {
  const std::ifstream input(path, std::ios::binary);
  if (!input) {
    SetError(error, "cannot open evaluation data");
    return std::nullopt;
  }
  std::ostringstream contents;
  contents << input.rdbuf();

  try {
    const Value root = gufo::server::json::parse(contents.str());
    if (!root.is_object() ||
        RequireString(root, "schema") != "gufo.eval-cases.v2") {
      throw std::runtime_error("unsupported evaluation schema");
    }

    EvalSuite suite;
    suite.benchmark = RequireString(root, "benchmark");
    suite.source_repository = RequireString(root, "source_repository");
    suite.source_revision = RequireString(root, "source_revision");
    suite.source_path = RequireString(root, "source_path");
    suite.source_blob = RequireString(root, "source_blob");
    suite.system_prompt = RequireString(root, "system_prompt");

    const Value& cases = RequireMember(root, "cases", Value::Type::kArray);
    suite.cases.reserve(cases.size());
    for (const auto& value : cases.items()) {
      suite.cases.push_back(ParseCase(value));
    }
    if (suite.cases.empty()) {
      throw std::runtime_error("evaluation suite is empty");
    }
    return suite;
  } catch (const std::exception& exception) {
    SetError(error,
             "invalid evaluation data: " + std::string(exception.what()));
    return std::nullopt;
  }
}

std::string BuildUserPrompt(const EvalCase& eval_case) {
  std::ostringstream prompt;
  prompt << eval_case.question << '\n';
  if (eval_case.kind == AnswerKind::kMultipleChoice) {
    prompt << "\nChoices:\n";
    for (std::size_t index = 0; index < eval_case.choices.size(); ++index) {
      prompt << static_cast<char>('A' + index) << ". "
             << eval_case.choices[index] << '\n';
    }
    prompt << "\nSolve the question. At the end, write exactly one final line "
              "in this format and do not write anything after it:\n"
              "Answer: <letter>";
  } else if (eval_case.kind == AnswerKind::kLineSpec) {
    prompt << "\nAt the end, write exactly one final line in this format and "
              "do not write anything after it:\n"
              "Answer: <line number or comma-separated line numbers>";
  } else {
    prompt << "\nSolve the problem. At the end, write exactly one final line "
              "in this format and do not write anything after it:\n"
              "Answer: <integer>";
  }
  return prompt.str();
}

std::string_view AnswerKindName(AnswerKind kind) noexcept {
  switch (kind) {
    case AnswerKind::kMultipleChoice:
      return "mcq";
    case AnswerKind::kInteger:
      return "integer";
    case AnswerKind::kLineSpec:
      return "linespec";
  }
  return "unknown";
}

}  // namespace gufo::eval
