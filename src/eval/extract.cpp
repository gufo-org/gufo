#include "src/eval/extract.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace gufo::eval {
namespace {

std::string Lower(std::string_view value) {
  std::string result(value);
  std::ranges::transform(result, result.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return result;
}

bool IsWordCharacter(char character) {
  return std::isalnum(static_cast<unsigned char>(character)) != 0 ||
         character == '_';
}

std::string_view VisibleText(std::string_view text) {
  const std::size_t marker = text.rfind("</think>");
  return marker == std::string_view::npos ? text : text.substr(marker + 8);
}

std::optional<std::size_t> LastAnswerMarker(std::string_view text) {
  const std::string lower = Lower(text);
  std::optional<std::size_t> fallback;
  std::optional<std::size_t> with_colon;
  std::size_t position = 0;
  while ((position = lower.find("answer", position)) != std::string::npos) {
    const bool left_boundary =
        position == 0 || !IsWordCharacter(lower[position - 1]);
    const std::size_t after = position + 6;
    const bool right_boundary =
        after == lower.size() || !IsWordCharacter(lower[after]);
    if (left_boundary && right_boundary) {
      fallback = position;
      std::size_t cursor = after;
      while (cursor < lower.size() &&
             std::isspace(static_cast<unsigned char>(lower[cursor])) != 0 &&
             lower[cursor] != '\n') {
        ++cursor;
      }
      if (cursor < lower.size() && lower[cursor] == ':') {
        with_colon = position;
      }
    }
    position = after;
  }
  return with_colon.has_value() ? with_colon : fallback;
}

std::string_view AnswerWindow(std::string_view text, std::size_t limit) {
  const std::string_view visible = VisibleText(text);
  const auto marker = LastAnswerMarker(visible);
  if (!marker.has_value()) {
    return visible;
  }
  std::string_view window = visible.substr(*marker);
  const std::size_t newline = window.find('\n');
  if (newline != std::string_view::npos) {
    window = window.substr(0, newline);
  }
  return window.substr(0, std::min(window.size(), limit));
}

bool IsLetterBoundary(char character) {
  return character == '\0' ||
         std::isalnum(static_cast<unsigned char>(character)) == 0;
}

bool HasNegationCue(std::string_view text, std::size_t letter_position) {
  std::size_t end = letter_position;
  while (end > 0 && (text[end - 1] == ' ' || text[end - 1] == '\t' ||
                     text[end - 1] == ',' || text[end - 1] == ';')) {
    --end;
  }
  std::size_t start = end;
  while (start > 0 &&
         (std::isalpha(static_cast<unsigned char>(text[start - 1])) != 0 ||
          text[start - 1] == '\'')) {
    --start;
  }
  const std::string word = Lower(text.substr(start, end - start));
  if (word.ends_with("n't")) {
    return true;
  }
  static const std::set<std::string> kCues = {
      "not",      "except",    "excluding",  "exclude",
      "excludes", "eliminate", "eliminates", "eliminated",
      "reject",   "rejects",   "rejected",   "rejecting",
  };
  if (kCues.contains(word)) {
    return true;
  }
  if (word != "out") {
    return false;
  }
  end = start;
  while (end > 0 && (text[end - 1] == ' ' || text[end - 1] == '\t')) {
    --end;
  }
  start = end;
  while (start > 0 &&
         std::isalpha(static_cast<unsigned char>(text[start - 1])) != 0) {
    --start;
  }
  const std::string rule = Lower(text.substr(start, end - start));
  return rule == "rule" || rule == "rules" || rule == "ruled";
}

bool LooksLikeLeadingProse(std::string_view text, std::size_t position,
                           char candidate) {
  if (candidate != 'A' && candidate != 'I') {
    return false;
  }
  if (position + 1 >= text.size()) {
    return false;
  }
  if (text[position + 1] == '\'') {
    return true;
  }
  std::size_t cursor = position + 1;
  if (text[cursor] != ' ' && text[cursor] != '\t') {
    return false;
  }
  while (cursor < text.size() &&
         (text[cursor] == ' ' || text[cursor] == '\t')) {
    ++cursor;
  }
  return cursor < text.size() &&
         std::islower(static_cast<unsigned char>(text[cursor])) != 0;
}

ExtractionResult ExtractMultipleChoice(const EvalCase& eval_case,
                                       std::string_view text) {
  const std::string_view visible = VisibleText(text);
  const auto marker = LastAnswerMarker(visible);
  const std::string_view window =
      marker.has_value() ? AnswerWindow(visible, 160) : visible;
  const char maximum = static_cast<char>('A' + eval_case.choices.size() - 1);
  std::vector<std::pair<char, std::size_t>> candidates;

  for (std::size_t index = 0; index < window.size(); ++index) {
    const char candidate = static_cast<char>(
        std::toupper(static_cast<unsigned char>(window[index])));
    if (candidate < 'A' || candidate > maximum) {
      continue;
    }
    const char before = index == 0 ? '\0' : window[index - 1];
    const char after = index + 1 == window.size() ? '\0' : window[index + 1];
    if (!IsLetterBoundary(before) || !IsLetterBoundary(after) ||
        LooksLikeLeadingProse(window, index, candidate) ||
        HasNegationCue(window, index)) {
      continue;
    }
    candidates.emplace_back(candidate, index);
  }
  if (candidates.empty()) {
    return {};
  }

  const char selected =
      marker.has_value() ? candidates.front().first : candidates.back().first;
  for (const auto& [candidate, position] : candidates) {
    if (candidate == selected) {
      continue;
    }
    const std::size_t selected_position =
        marker.has_value() ? candidates.front().second : position;
    const std::size_t begin = std::min(selected_position, position);
    const std::size_t end = std::max(selected_position, position);
    const std::string between = Lower(window.substr(begin, end - begin));
    if (between.find(" or ") != std::string::npos ||
        between.find('/') != std::string::npos ||
        between.find(" and ") != std::string::npos) {
      return {.status = ExtractionStatus::kAmbiguous, .answer = ""};
    }
  }
  return {.status = ExtractionStatus::kFound,
          .answer = std::string(1, selected)};
}

std::string NormalizeInteger(std::string value) {
  std::erase(value, ',');
  std::erase(value, '$');
  bool negative = false;
  std::size_t start = 0;
  if (!value.empty() && (value.front() == '-' || value.front() == '+')) {
    negative = value.front() == '-';
    start = 1;
  }
  while (start + 1 < value.size() && value[start] == '0') {
    ++start;
  }
  std::string normalized = value.substr(start);
  if (negative && normalized != "0") {
    normalized.insert(normalized.begin(), '-');
  }
  return normalized;
}

std::vector<std::pair<std::string, std::size_t>> Integers(
    std::string_view text) {
  std::vector<std::pair<std::string, std::size_t>> values;
  for (std::size_t index = 0; index < text.size();) {
    const std::size_t start = index;
    if ((text[index] == '-' || text[index] == '+') && index + 1 < text.size() &&
        std::isdigit(static_cast<unsigned char>(text[index + 1])) != 0) {
      ++index;
    }
    if (std::isdigit(static_cast<unsigned char>(text[index])) == 0) {
      ++index;
      continue;
    }
    while (index < text.size() &&
           (std::isdigit(static_cast<unsigned char>(text[index])) != 0 ||
            text[index] == ',')) {
      ++index;
    }
    values.emplace_back(
        NormalizeInteger(std::string(text.substr(start, index - start))),
        start);
  }
  return values;
}

ExtractionResult ExtractInteger(std::string_view text) {
  const std::string_view visible = VisibleText(text);
  const auto marker = LastAnswerMarker(visible);
  std::string_view window =
      marker.has_value() ? AnswerWindow(visible, 200) : visible;
  const std::size_t equals = window.rfind('=');
  if (equals != std::string_view::npos) {
    window = window.substr(equals + 1);
  }
  const auto values = Integers(window);
  if (values.empty()) {
    return {};
  }
  if (values.size() > 1) {
    const std::string lower = Lower(window);
    if (lower.find(" or ") != std::string::npos ||
        lower.find('/') != std::string::npos) {
      return {.status = ExtractionStatus::kAmbiguous, .answer = ""};
    }
  }
  const std::string& answer =
      marker.has_value() || equals != std::string_view::npos
          ? values.front().first
          : values.back().first;
  return {.status = ExtractionStatus::kFound, .answer = answer};
}

std::vector<int> ParseLineSet(std::string_view text) {
  std::set<int> lines;
  for (std::size_t index = 0; index < text.size();) {
    if (std::isdigit(static_cast<unsigned char>(text[index])) == 0) {
      ++index;
      continue;
    }
    int first = 0;
    const char* begin = text.data() + index;
    const char* end = text.data() + text.size();
    const auto first_parse = std::from_chars(begin, end, first);
    if (first_parse.ec != std::errc{}) {
      ++index;
      continue;
    }
    index = static_cast<std::size_t>(first_parse.ptr - text.data());
    int last = first;
    std::size_t cursor = index;
    while (cursor < text.size() &&
           std::isspace(static_cast<unsigned char>(text[cursor])) != 0) {
      ++cursor;
    }
    if (cursor < text.size() && text[cursor] == '-') {
      ++cursor;
      while (cursor < text.size() &&
             std::isspace(static_cast<unsigned char>(text[cursor])) != 0) {
        ++cursor;
      }
      const auto last_parse = std::from_chars(text.data() + cursor, end, last);
      if (last_parse.ec == std::errc{}) {
        index = static_cast<std::size_t>(last_parse.ptr - text.data());
      } else {
        last = first;
      }
    }
    if (first > last) {
      std::swap(first, last);
    }
    for (int line = first; line <= last && line < 100000; ++line) {
      lines.insert(line);
    }
  }
  return {lines.begin(), lines.end()};
}

std::string FormatLineSet(const std::vector<int>& lines) {
  std::string output;
  for (const int line : lines) {
    if (!output.empty()) {
      output.push_back(',');
    }
    output += std::to_string(line);
  }
  return output;
}

ExtractionResult ExtractLineSpec(std::string_view text) {
  const std::string_view window = AnswerWindow(text, 240);
  const auto lines = ParseLineSet(window);
  if (lines.empty()) {
    return {};
  }
  return {.status = ExtractionStatus::kFound, .answer = FormatLineSet(lines)};
}

}  // namespace

ExtractionResult ExtractAnswer(const EvalCase& eval_case,
                               std::string_view text) {
  switch (eval_case.kind) {
    case AnswerKind::kMultipleChoice:
      return ExtractMultipleChoice(eval_case, text);
    case AnswerKind::kInteger:
      return ExtractInteger(text);
    case AnswerKind::kLineSpec:
      return ExtractLineSpec(text);
  }
  return {};
}

bool AnswerMatches(const EvalCase& eval_case, std::string_view extracted) {
  if (eval_case.kind == AnswerKind::kLineSpec) {
    const auto expected = ParseLineSet(eval_case.answer);
    const auto actual = ParseLineSet(extracted);
    if (actual.empty()) {
      return false;
    }
    return std::ranges::all_of(actual, [&expected](int line) {
      return std::ranges::find(expected, line) != expected.end();
    });
  }
  if (eval_case.kind == AnswerKind::kInteger) {
    return NormalizeInteger(std::string(extracted)) ==
           NormalizeInteger(eval_case.answer);
  }
  return extracted == eval_case.answer;
}

GradeResult GradeAnswer(const EvalCase& eval_case, std::string_view text) {
  GradeResult result;
  result.extraction = ExtractAnswer(eval_case, text);
  result.passed = result.extraction.status == ExtractionStatus::kFound &&
                  AnswerMatches(eval_case, result.extraction.answer);
  return result;
}

std::string_view ExtractionStatusName(ExtractionStatus status) noexcept {
  switch (status) {
    case ExtractionStatus::kFound:
      return "found";
    case ExtractionStatus::kNoAnswer:
      return "no_answer";
    case ExtractionStatus::kAmbiguous:
      return "ambiguous";
  }
  return "unknown";
}

}  // namespace gufo::eval
