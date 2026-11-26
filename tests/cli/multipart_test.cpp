// Checks ParseMultipart, the C++ caller of the Rust multipart/form-data parser
// (src/core/http/src/multipart.rs), with each API's limits: the forms captured
// from curl, Python requests, Node.js and Go, stopping after a chosen field,
// every rejection with its status and detail, and exceptions from the field
// consumer. The parser's Rust unit tests check the RFC 7578, RFC 2046 and
// RFC 9110 rules themselves, including where 0.33.0 read forms differently
// (see TestDifferences).
#include "src/cli/serve/multipart.hpp"

#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using gufo::server::MultipartField;
using gufo::server::MultipartLimits;
using gufo::server::MultipartStatus;

constexpr std::size_t kNever = std::numeric_limits<std::size_t>::max();

// As audio_asr_api.cpp, video_api.cpp and image_api.cpp set them.
constexpr MultipartLimits kAsr{.max_fields = 12, .unique_names = true};
constexpr MultipartLimits kVideo{.max_fields = 16, .unique_names = true};
constexpr MultipartLimits kImage{.max_fields = 32, .unique_names = false};

struct Field {
  std::string name;
  bool has_filename = false;
  std::string content;
  bool operator==(const Field&) const = default;
};

struct Outcome {
  MultipartStatus status;
  std::string detail;
  // The fields the consumer was called with, the stopping one included.
  std::vector<Field> fields;
  bool operator==(const Outcome&) const = default;
};

Outcome Parse(std::string_view content_type, std::string_view body,
              MultipartLimits limits, std::size_t stop_at = kNever) {
  Outcome outcome;
  auto form = gufo::server::ParseMultipart(
      content_type, body, limits, [&](const MultipartField& field) {
        outcome.fields.push_back({.name = std::string(field.name),
                                  .has_filename = field.has_filename,
                                  .content = std::string(field.content)});
        return outcome.fields.size() != stop_at + 1;
      });
  outcome.status = form.status;
  outcome.detail = std::move(form.detail);
  return outcome;
}

int failures = 0;

void Expect(bool ok, std::string_view what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %.*s\n", static_cast<int>(what.size()),
                 what.data());
    ++failures;
  }
}

std::string Bytes(int from, int to) {
  std::string bytes;
  for (int byte = from; byte < to; ++byte) {
    bytes += static_cast<char>(byte);
  }
  return bytes;
}

// What every client sent (tests/fixtures/multipart/README.md).
std::vector<Field> CapturedFields() {
  const std::string png("\x89PNG\r\n\x1a\n", 8);
  return {
      {.name = "model", .has_filename = false, .content = "whisper-1"},
      {.name = "prompt",
       .has_filename = false,
       .content = "Gr\xc3\xbc\xc3\x9f"
                  "e \"quoted\""},
      {.name = "file",
       .has_filename = true,
       .content = "RIFF" + Bytes(0, 256) + "\r\n--x\r\n"},
      {.name = "image[]", .has_filename = true, .content = png + Bytes(0, 64)},
      {.name = "image[]",
       .has_filename = true,
       .content = png + Bytes(64, 128)},
  };
}

// Each captured form gives the image API its five fields, and also stops
// after each of the first three; ASR and video reject the repeated image[].
void TestClients() {
  const std::vector<Field> expected = CapturedFields();
  std::size_t clients = 0;
  for (const auto& entry :
       std::filesystem::directory_iterator(GUFO_MULTIPART_FIXTURES)) {
    if (entry.path().extension() != ".form") {
      continue;
    }
    ++clients;
    const std::string client = entry.path().filename().string();
    std::ifstream file(entry.path(), std::ios::binary);
    const std::string form{std::istreambuf_iterator<char>(file), {}};
    constexpr std::string_view kHeader = "Content-Type: ";
    const std::size_t split = form.find("\r\n\r\n");
    if (!form.starts_with(kHeader) || split == std::string::npos) {
      Expect(false, client + " has no Content-Type line");
      continue;
    }
    const std::string_view text(form);
    const std::string_view content_type =
        text.substr(kHeader.size(), split - kHeader.size());
    const std::string_view body = text.substr(split + 4);

    Expect(Parse(content_type, body, kImage) ==
               Outcome{MultipartStatus::kComplete, "", expected},
           client + ": image API reads the captured fields");
    for (std::size_t stop = 0; stop < 3; ++stop) {
      const std::vector<Field> seen(
          expected.begin(),
          expected.begin() + static_cast<std::ptrdiff_t>(stop + 1));
      Expect(Parse(content_type, body, kImage, stop) ==
                 Outcome{MultipartStatus::kStopped, "", seen},
             client + ": image API stops after field " + std::to_string(stop));
      Expect(Parse(content_type, body, kVideo, stop) ==
                 Outcome{MultipartStatus::kStopped, "", seen},
             client + ": video API stops after field " + std::to_string(stop));
    }
    for (const MultipartLimits limits : {kAsr, kVideo}) {
      Expect(Parse(content_type, body, limits) ==
                 Outcome{MultipartStatus::kDuplicateField,
                         "image[]",
                         {expected.begin(), expected.end() - 1}},
             client + ": unique names reject the second image[]");
    }
  }
  Expect(clients == 4, "four captured clients");
}

std::string Part(std::string_view headers, std::string_view content) {
  return "--B\r\n" + std::string(headers) + "\r\n\r\n" + std::string(content) +
         "\r\n";
}

std::string Form(std::string_view headers) {
  return Part(headers, "v") + "--B--";
}

std::string Fields(int count) {
  std::string body;
  for (int index = 0; index < count; ++index) {
    body += Part("Content-Disposition: form-data; name=\"f" +
                     std::to_string(index) + "\"",
                 "v");
  }
  return body + "--B--";
}

struct Rejection {
  std::string content_type;
  std::string body;
  MultipartLimits limits;
  MultipartStatus status;
  std::string detail;
};

// Every status and message ParseMultipart can return for a client's form;
// ASR answers kNotFormData and kBadBoundary alike, the others pass detail on.
void TestRejections() {
  constexpr std::string_view kType = "multipart/form-data; boundary=B";
  const std::string name = "Content-Disposition: form-data; name=\"a\"";
  const std::vector<Rejection> rejections = {
      {"multipart/mixed; boundary=B", Form(name), kImage,
       MultipartStatus::kNotFormData, ""},
      {"", Form(name), kImage, MultipartStatus::kNotFormData, ""},
      {"multipart/form-data", Form(name), kImage, MultipartStatus::kBadBoundary,
       ""},
      {std::string(kType), "preamble\r\n" + Form(name), kImage,
       MultipartStatus::kMalformed, "multipart body has no opening boundary"},
      {std::string(kType), Part(name, "v") + "--B", kImage,
       MultipartStatus::kMalformed, "multipart body has no closing boundary"},
      // Neither 0.33.0 ASR nor video rejected a header line without a colon,
      // with an empty name or with a control byte; ASR dropped the byte.
      {std::string(kType), Form("NoColon\r\n" + name), kImage,
       MultipartStatus::kMalformed, "multipart part headers are malformed"},
      {std::string(kType), Form(": empty name\r\n" + name), kImage,
       MultipartStatus::kMalformed, "multipart part headers are malformed"},
      {std::string(kType),
       Form("Content-Disposition: form-data; name=\"a\x01"
            "b\""),
       kImage, MultipartStatus::kMalformed,
       "multipart part headers are malformed"},
      {std::string(kType),
       Form("X-Pad: " + std::string(16340, 'p') + "\r\n" + name), kImage,
       MultipartStatus::kMalformed,
       "multipart part headers exceed 16384 bytes"},
      {std::string(kType), Form("Content-Type: text/plain"), kImage,
       MultipartStatus::kMalformed,
       "multipart part needs exactly one Content-Disposition"},
      {std::string(kType), Form("Content-Disposition: attachment; name=\"a\""),
       kImage, MultipartStatus::kMalformed,
       "multipart part has an invalid Content-Disposition"},
      {std::string(kType), Form("Content-Disposition: form-data"), kImage,
       MultipartStatus::kMalformed, "multipart part has no field name"},
      {std::string(kType), Fields(13), kAsr, MultipartStatus::kTooManyFields,
       ""},
      {std::string(kType), Fields(17), kVideo, MultipartStatus::kTooManyFields,
       ""},
      {std::string(kType), Fields(33), kImage, MultipartStatus::kTooManyFields,
       ""},
  };
  for (const Rejection& rejection : rejections) {
    const Outcome outcome =
        Parse(rejection.content_type, rejection.body, rejection.limits);
    Expect(outcome.status == rejection.status &&
               outcome.detail == rejection.detail,
           "rejects \"" + rejection.content_type + "\" with \"" +
               rejection.detail + "\" (got \"" + outcome.detail + "\")");
  }
  // Each API's field limit itself is accepted.
  Expect(Parse(kType, Fields(12), kAsr).status == MultipartStatus::kComplete,
         "ASR reads 12 fields");
  Expect(Parse(kType, Fields(16), kVideo).status == MultipartStatus::kComplete,
         "video reads 16 fields");
  Expect(Parse(kType, Fields(32), kImage).fields.size() == 32,
         "image reads 32 fields");
}

// Where 0.33.0 read a form differently from RFC 7578. The Rust unit tests
// check these rules in this form; TestRejections checks header lines without
// a colon, with an empty name or with a control byte, and this test a quoted
// name between tabs:
// - a boundary is 1 to 70 RFC 2046 characters (0.33.0 video took 71, and
//   every API a non-ASCII one), and only one boundary parameter (ASR and
//   image used the first);
// - media type and parameter names are case-insensitive (image rejected
//   "Multipart/Form-Data", "Boundary", "NAME" and "Form-Data");
// - "multipart/form-dataX" and "multipart/mixed" are not forms (video and
//   image read them);
// - a delimiter may be followed by spaces and tabs (all rejected "--B \r\n"),
//   and content may hold "\r\n--Bx" (ASR and video split there);
// - an epilogue after the close delimiter is ignored (video and image
//   rejected the form after reading its fields);
// - parameters are RFC 9110 quoted strings, each at most once, with blanks
//   around them (ASR read `filename="a;name=x"` as name `x"` and kept
//   escapes; ASR and image took the first of two names; ASR read `name="a`
//   and attachment parts; image kept the quotes of `name=\t"a"\t`);
// - a header name is followed directly by its colon (ASR and video accepted
//   "Content-Disposition :" and leading spaces), and a part has exactly one
//   Content-Disposition (ASR took the last);
// - "filename*" marks a file (video only saw "filename").
void TestDifferences() {
  Expect(
      Parse("multipart/form-data; boundary=B",
            Form("Content-Disposition: form-data; name=\t\"a\"\t"), kImage) ==
          Outcome{MultipartStatus::kComplete, "", {{"a", false, "v"}}},
      "reads a quoted name between tabs");
}

// A field consumer's exception leaves ParseMultipart unchanged.
void TestExceptions() {
  bool propagated = false;
  try {
    gufo::server::ParseMultipart(
        "multipart/form-data; boundary=B",
        Form("Content-Disposition: form-data; name=\"a\""), kImage,
        [](const MultipartField&) -> bool {
          throw std::invalid_argument("consumer");
        });
  } catch (const std::invalid_argument& error) {
    propagated = std::string_view(error.what()) == "consumer";
  }
  Expect(propagated, "a field consumer's exception propagates");
}

}  // namespace

int main() {
  try {
    TestClients();
    TestRejections();
    TestDifferences();
    TestExceptions();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
  if (failures != 0) {
    std::fprintf(stderr, "%d multipart checks failed\n", failures);
    return 1;
  }
  std::printf("multipart checks passed\n");
  return 0;
}
