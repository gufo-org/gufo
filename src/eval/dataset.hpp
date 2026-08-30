#ifndef GUFO_EVAL_DATASET_HPP_
#define GUFO_EVAL_DATASET_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gufo::eval {

enum class AnswerKind : std::uint8_t {
  kMultipleChoice,
  kInteger,
  kLineSpec,
};

struct EvalCase {
  std::size_t ds4_index{0};
  std::string source;
  std::string id;
  std::string domain;
  std::string title;
  AnswerKind kind{AnswerKind::kMultipleChoice};
  std::string question;
  std::vector<std::string> choices;
  std::string answer;
  std::string dataset;
  std::string dataset_url;
  std::string license;
  std::string audit_note;
  std::string source_record_sha256;
};

struct EvalSuite {
  std::string benchmark;
  std::string source_repository;
  std::string source_revision;
  std::string source_path;
  std::string source_blob;
  std::string system_prompt;
  std::vector<EvalCase> cases;
};

[[nodiscard]] std::optional<EvalSuite> LoadEvalSuite(
    const std::filesystem::path& path, std::string* error = nullptr);

[[nodiscard]] std::string BuildUserPrompt(const EvalCase& eval_case);

[[nodiscard]] std::string_view AnswerKindName(AnswerKind kind) noexcept;

}  // namespace gufo::eval

#endif  // GUFO_EVAL_DATASET_HPP_
