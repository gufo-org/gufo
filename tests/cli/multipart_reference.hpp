#ifndef GUFO_TESTS_CLI_MULTIPART_REFERENCE_HPP_
#define GUFO_TESTS_CLI_MULTIPART_REFERENCE_HPP_

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::multipart_reference {

struct Field {
  std::string name;
  bool has_filename = false;
  std::string content;

  bool operator==(const Field&) const = default;
};

enum class Status { kComplete, kContentType, kMalformed, kStopped };

using FieldFn = std::function<bool(const Field&)>;

// What 0.33.0 made of a form: its fields in body order, or why it was
// rejected and the message the client got.
struct Result {
  Status status;
  std::string message;
  std::vector<Field> fields;
};

// ASR keeps no fields of a rejected form; video and image hand over each
// field as they reach it, and stop after the one at index stop_at.
Result Asr(std::string_view content_type, std::string_view body);
Result Video(std::string_view content_type, std::string_view body,
             std::size_t stop_at);
Result Image(std::string_view content_type, std::string_view body,
             std::size_t stop_at);

}  // namespace gufo::multipart_reference

#endif  // GUFO_TESTS_CLI_MULTIPART_REFERENCE_HPP_
