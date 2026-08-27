#ifndef GUFO_EVAL_EXTRACT_HPP_
#define GUFO_EVAL_EXTRACT_HPP_

#include <cstdint>
#include <string>
#include <string_view>

#include "src/eval/dataset.hpp"

namespace gufo::eval {

enum class ExtractionStatus : std::uint8_t {
  kFound,
  kNoAnswer,
  kAmbiguous,
};

struct ExtractionResult {
  ExtractionStatus status{ExtractionStatus::kNoAnswer};
  std::string answer;
};

struct GradeResult {
  ExtractionResult extraction;
  bool passed{false};
};

[[nodiscard]] ExtractionResult ExtractAnswer(const EvalCase& eval_case,
                                             std::string_view text);

[[nodiscard]] bool AnswerMatches(const EvalCase& eval_case,
                                 std::string_view extracted);

[[nodiscard]] GradeResult GradeAnswer(const EvalCase& eval_case,
                                      std::string_view text);

[[nodiscard]] std::string_view ExtractionStatusName(
    ExtractionStatus status) noexcept;

}  // namespace gufo::eval

#endif  // GUFO_EVAL_EXTRACT_HPP_
