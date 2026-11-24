#ifndef GUFO_SERVER_MULTIPART_HPP_
#define GUFO_SERVER_MULTIPART_HPP_

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

namespace gufo::server {

struct MultipartField {
  std::string_view name;
  // Its Content-Disposition has a filename (or filename*) parameter.
  bool has_filename;
  std::string_view content;
};

// What an API allows.
struct MultipartLimits {
  std::size_t max_fields;
  bool unique_names;
};

enum class MultipartStatus {
  kComplete,
  kNotFormData,
  // The boundary is missing, not 1 to 70 RFC 2046 characters, or malformed.
  kBadBoundary,
  // The body breaks RFC 7578; detail says how.
  kMalformed,
  kTooManyFields,
  // detail is the repeated name.
  kDuplicateField,
  // on_field returned false.
  kStopped,
};

struct MultipartResult {
  MultipartStatus status;
  std::string detail;
};

// Reads a multipart/form-data body (src/core/http/src/multipart.rs), calling
// on_field for each field in order until it returns false; its views live
// during the call only. An exception from on_field propagates; a failure of
// the parser itself throws std::runtime_error.
MultipartResult ParseMultipart(
    std::string_view content_type, std::string_view body,
    MultipartLimits limits,
    const std::function<bool(const MultipartField&)>& on_field);

}  // namespace gufo::server

#endif  // GUFO_SERVER_MULTIPART_HPP_
