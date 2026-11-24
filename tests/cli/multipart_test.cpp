// Compares the Rust multipart/form-data parser, through ParseMultipart with
// each API's limits, with frozen copies of the three 0.33.0 parsers
// (tests/cli/multipart_reference.cpp). Forms as clients encode them (see
// Canonical) must give the same result in every API: accepted with the same
// fields in the same order, or rejected; error messages may differ. Other
// forms may differ, as RFC 7578 reads them; the test counts them. Forms
// captured from curl, Python requests, Node.js and Go are among the canonical
// ones. Video and image also stop after a chosen field, where their field
// consumers reject one.
#include "src/cli/serve/multipart.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "tests/cli/multipart_reference.hpp"

namespace {

using gufo::server::MultipartField;
using gufo::server::MultipartLimits;
using gufo::server::MultipartStatus;
namespace reference = gufo::multipart_reference;

constexpr std::size_t kNever = std::numeric_limits<std::size_t>::max();

enum class Api { kAsr, kVideo, kImage };
constexpr Api kApis[] = {Api::kAsr, Api::kVideo, Api::kImage};

const char* Name(Api api) {
  return api == Api::kAsr ? "asr" : api == Api::kVideo ? "video" : "image";
}

// What an API does with a form: the fields it reads (ASR only keeps those of
// an accepted form; only video looks at filenames).
struct Outcome {
  enum class Status { kAccepted, kStopped, kRejected } status;
  std::vector<reference::Field> fields;
  bool operator==(const Outcome&) const = default;
};

Outcome Old(Api api, std::string_view content_type, std::string_view body,
            std::size_t stop_at) {
  const reference::Result result =
      api == Api::kAsr     ? reference::Asr(content_type, body)
      : api == Api::kVideo ? reference::Video(content_type, body, stop_at)
                           : reference::Image(content_type, body, stop_at);
  return {.status = result.status == reference::Status::kComplete
                        ? Outcome::Status::kAccepted
                    : result.status == reference::Status::kStopped
                        ? Outcome::Status::kStopped
                        : Outcome::Status::kRejected,
          .fields = result.fields};
}

MultipartLimits Limits(Api api) {
  return api == Api::kAsr     ? MultipartLimits{12, true}
         : api == Api::kVideo ? MultipartLimits{16, true}
                              : MultipartLimits{32, false};
}

// Every status and message the new parser gave.
std::set<std::string> reached;

Outcome New(Api api, std::string_view content_type, std::string_view body,
            std::size_t stop_at) {
  Outcome outcome{.status = Outcome::Status::kAccepted, .fields = {}};
  const auto form = gufo::server::ParseMultipart(
      content_type, body, Limits(api), [&](const MultipartField& field) {
        outcome.fields.push_back(
            {.name = std::string(field.name),
             .has_filename = api == Api::kVideo && field.has_filename,
             .content = std::string(field.content)});
        return outcome.fields.size() != stop_at + 1;
      });
  switch (form.status) {
    case MultipartStatus::kComplete:
      break;
    case MultipartStatus::kStopped:
      outcome.status = Outcome::Status::kStopped;
      break;
    case MultipartStatus::kMalformed:
      reached.insert(form.detail);
      outcome.status = Outcome::Status::kRejected;
      break;
    default:
      reached.insert("status " + std::to_string(static_cast<int>(form.status)));
      outcome.status = Outcome::Status::kRejected;
      break;
  }
  if (api == Api::kAsr && outcome.status != Outcome::Status::kAccepted) {
    outcome.fields.clear();
  }
  return outcome;
}

bool Tchar(unsigned char c) {
  return std::isalnum(c) != 0 ||
         (c != 0 && std::strchr("!#$%&'*+-.^_`|~", c) != nullptr);
}

// Text in a quoted parameter as clients write it: no quote, backslash or
// control byte, and for names no semicolon.
bool PlainText(std::string_view text, bool name) {
  return std::ranges::none_of(text, [name](char byte) {
    const auto c = static_cast<unsigned char>(byte);
    return c < 0x20 || c == 0x7f || c == '"' || c == '\\' || (name && c == ';');
  });
}

bool CanonicalDisposition(std::string_view value) {
  constexpr std::string_view kName = "form-data; name=\"";
  if (!value.starts_with(kName)) {
    return false;
  }
  const std::size_t name_end = value.find('"', kName.size());
  if (name_end == std::string_view::npos || name_end == kName.size() ||
      !PlainText(value.substr(kName.size(), name_end - kName.size()), true)) {
    return false;
  }
  const std::string_view rest = value.substr(name_end + 1);
  constexpr std::string_view kFilename = "; filename=\"";
  return rest.empty() ||
         (rest.starts_with(kFilename) && rest.size() > kFilename.size() &&
          rest.ends_with('"') &&
          PlainText(
              rest.substr(kFilename.size(), rest.size() - kFilename.size() - 1),
              false));
}

// A form as curl, Python requests, browsers (FormData) and Go encode it:
// Content-Type "multipart/form-data; boundary=B" (B quoted or not, 1 to 70
// RFC 2046 characters without space), "--B" CRLF, each part's header lines
// "Name: value" with exactly one Content-Disposition
// `form-data; name="N"[; filename="F"]`, an empty line, content without CRLF
// "--B", CRLF "--B", and "--" with an optional CRLF at the end.
bool Canonical(std::string_view content_type, std::string_view body) {
  constexpr std::string_view kPrefix = "multipart/form-data; boundary=";
  if (!content_type.starts_with(kPrefix)) {
    return false;
  }
  std::string_view boundary = content_type.substr(kPrefix.size());
  if (boundary.size() >= 2 && boundary.front() == '"' &&
      boundary.back() == '"') {
    boundary = boundary.substr(1, boundary.size() - 2);
  }
  if (boundary.empty() || boundary.size() > 70 ||
      !std::ranges::all_of(boundary, [](char byte) {
        const auto c = static_cast<unsigned char>(byte);
        return std::isalnum(c) != 0 ||
               (c != 0 && std::strchr("'()+_,-./:=?", c) != nullptr);
      })) {
    return false;
  }
  const std::string dash = "--" + std::string(boundary);
  const std::string delimiter = "\r\n" + dash;
  if (body == dash + "--" || body == dash + "--\r\n") {
    return true;
  }
  if (!body.starts_with(dash + "\r\n")) {
    return false;
  }
  std::size_t at = dash.size() + 2;
  while (true) {
    const std::size_t headers_end = body.find("\r\n\r\n", at);
    if (headers_end == std::string_view::npos || headers_end - at > 1000) {
      return false;
    }
    int dispositions = 0;
    std::string_view headers = body.substr(at, headers_end - at);
    while (true) {
      const std::size_t line_end = headers.find("\r\n");
      const std::string_view line = headers.substr(0, line_end);
      const std::size_t colon = line.find(": ");
      if (colon == std::string_view::npos || colon == 0 ||
          !std::ranges::all_of(line.substr(0, colon),
                               [](char c) { return Tchar(c); })) {
        return false;
      }
      const std::string_view value = line.substr(colon + 2);
      if (value.empty() || value.front() == ' ' || value.back() == ' ' ||
          std::ranges::any_of(value, [](char byte) {
            const auto c = static_cast<unsigned char>(byte);
            return c < 0x20 || c == 0x7f;
          })) {
        return false;
      }
      std::string name(line.substr(0, colon));
      std::ranges::transform(name, name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      if (name == "content-disposition") {
        ++dispositions;
        if (!CanonicalDisposition(value)) {
          return false;
        }
      }
      if (line_end == std::string_view::npos) {
        break;
      }
      headers.remove_prefix(line_end + 2);
    }
    // The CRLF that ends the headers must not start a delimiter either.
    const std::size_t end = body.find(delimiter, headers_end + 2);
    if (dispositions != 1 || end == std::string_view::npos ||
        end < headers_end + 4) {
      return false;
    }
    at = end + delimiter.size();
    const std::string_view rest = body.substr(at);
    if (rest == "--" || rest == "--\r\n") {
      return true;
    }
    if (!rest.starts_with("\r\n")) {
      return false;
    }
    at += 2;
  }
}

std::string Printable(std::string_view bytes) {
  std::string out;
  for (const unsigned char byte : bytes.substr(0, 300)) {
    if (byte >= 0x20 && byte < 0x7f && byte != '\\') {
      out += static_cast<char>(byte);
    } else {
      char escaped[8];
      std::snprintf(escaped, sizeof(escaped), "\\x%02x", byte);
      out += escaped;
    }
  }
  return out;
}

struct Totals {
  std::size_t canonical = 0;
  std::size_t accepted = 0;
  std::size_t failures = 0;
  std::size_t others = 0;
  // Other forms that some API now reads differently.
  std::size_t others_differing = 0;
};

void Fail(Totals& totals, Api api, std::size_t stop,
          std::string_view content_type, std::string_view body,
          const Outcome& old, const Outcome& now) {
  if (++totals.failures <= 12) {
    std::fprintf(stderr,
                 "%s differs (stop %zu): \"%s\" / \"%s\"\n  old: %d with %zu "
                 "fields, now: %d with %zu fields\n",
                 Name(api), stop, Printable(content_type).c_str(),
                 Printable(body).c_str(), static_cast<int>(old.status),
                 old.fields.size(), static_cast<int>(now.status),
                 now.fields.size());
  }
}

// A canonical form must give the same outcome in every API, also stopped
// after each of its first fields; another form is only counted.
void Compare(Totals& totals, std::string_view content_type,
             std::string_view body) {
  const bool canonical = Canonical(content_type, body);
  ++(canonical ? totals.canonical : totals.others);
  bool differs = false;
  for (const Api api : kApis) {
    const Outcome old = Old(api, content_type, body, kNever);
    const Outcome now = New(api, content_type, body, kNever);
    if (!canonical) {
      differs |= !(old == now);
      continue;
    }
    if (!(old == now)) {
      Fail(totals, api, kNever, content_type, body, old, now);
      continue;
    }
    if (old.status != Outcome::Status::kAccepted) {
      continue;
    }
    ++totals.accepted;
    for (std::size_t stop = 0;
         api != Api::kAsr && stop < std::min<std::size_t>(old.fields.size(), 3);
         ++stop) {
      const Outcome old_stopped = Old(api, content_type, body, stop);
      const Outcome now_stopped = New(api, content_type, body, stop);
      if (!(old_stopped == now_stopped)) {
        Fail(totals, api, stop, content_type, body, old_stopped, now_stopped);
      }
    }
  }
  totals.others_differing += differs ? 1 : 0;
}

template<typename T>
const T& Pick(std::mt19937_64& random, const std::vector<T>& values) {
  return values[std::uniform_int_distribution<std::size_t>(
      0, values.size() - 1)(random)];
}

bool Chance(std::mt19937_64& random, int percent) {
  return std::uniform_int_distribution<int>(0, 99)(random) < percent;
}

std::string Substitute(std::string text, std::string_view key,
                       std::string_view value) {
  for (std::size_t at = text.find(key); at != std::string::npos;
       at = text.find(key, at + value.size())) {
    text.replace(at, key.size(), value);
  }
  return text;
}

// Boundaries as clients make them, and more.
const std::vector<std::string> kBoundaries = {
    "B",
    "------------------------datqFTQ9OxnMi1eA5Eb2ts",
    "00ccc5cc2d535b854c0ba018bcb5ebf9",
    "----formdata-undici-066147008053",
    "e561e812e061b00f49f013201dcc0abd2d98580ac5f079bc2f8e18aa4eb8",
    "----WebKitFormBoundary7MA4YWxkTrZu0gW",
    "a'()+_,-./:=?z",
    std::string(70, 'b'),
    std::string(71, 'b'),
    std::string(200, 'b'),
    "a b",
    "x;y",
    "q\"z",
    "\xc3\xa4"};

const std::vector<std::string> kNames = {
    "file",        "model",  "prompt",     "language", "response_format",
    "temperature", "stream", "max_tokens", "gufo",     "first_frame",
    "last_frame",  "image",  "image[]",    "n",        "seed",
    "steps",       "size",   "x",          "a b",      "\xc3\xa4",
    "a=b"};

const std::vector<std::string> kFilenames = {
    "a.wav", "", "photo 1.png", "x;y.png", "\xc3\xa4.png", "a=b.png"};

const std::vector<std::string> kPartTypes = {"audio/wav", "image/png",
                                             "application/octet-stream",
                                             "text/plain; charset=utf-8"};

const std::vector<std::string> kContents = {
    "",
    "v",
    "whisper-1",
    "Gr\xc3\xbc\xc3\x9f"
    "e \"quoted\"",
    "{\"seed\": 4}",
    "4",
    "true",
    std::string("RIFF\x00\x01\r\n--x\r\n\xff", 12),
    "a--{B}",
    "a\r\n-",
    "line\r\nline",
    "\r\n",
    "a\r\n--{B}x",
    "a\r\n--{B} \r\n",
    "--{B}--",
};

// A form as a client encodes it, unless a content holds the boundary.
std::string ClientForm(std::mt19937_64& random, const std::string& boundary) {
  std::string body;
  const int count = Chance(random, 10)
                        ? std::uniform_int_distribution<int>(11, 34)(random)
                        : std::uniform_int_distribution<int>(0, 6)(random);
  for (int index = 0; index < count; ++index) {
    const std::string name =
        Chance(random, 15) ? "f" + std::to_string(index) : Pick(random, kNames);
    body += "--" + boundary + "\r\nContent-Disposition: form-data; name=\"" +
            name + "\"";
    if (Chance(random, 30)) {
      body += "; filename=\"" + Pick(random, kFilenames) + "\"";
    }
    body += "\r\n";
    if (Chance(random, 30)) {
      body += "Content-Type: " + Pick(random, kPartTypes) + "\r\n";
    }
    body += "\r\n" +
            Substitute(Chance(random, 70) ? kContents[1 + index % 7]
                                          : Pick(random, kContents),
                       "{B}", boundary) +
            "\r\n";
  }
  return body + "--" + boundary + (Chance(random, 50) ? "--\r\n" : "--");
}

// Content-Disposition lines around name {N} that clients do not send.
const std::vector<std::string> kDispositions = {
    "content-disposition: form-data; name={N}",
    "CONTENT-DISPOSITION:form-data;name=\"{N}\"",
    "Content-Disposition : form-data; name=\"{N}\"",
    " Content-Disposition: form-data; name=\"{N}\"",
    "Content-Disposition: form-data; filename=\"a;name=x\"; name=\"{N}\"",
    "Content-Disposition: form-data; NAME=\"{N}\"",
    "Content-Disposition: form-data; name=\"{N}\\\"q\"",
    "Content-Disposition: form-data; name=\"{N}\\",
    "Content-Disposition: form-data; name=\"\"",
    "Content-Disposition: form-data",
    "Content-Disposition: form-data; name=\"{N}\";",
    "Content-Disposition: attachment; name=\"{N}\"",
    "Content-Disposition: Form-Data; name=\"{N}\"",
    "Content-Disposition: form-data; name=\"{N}\"; name=\"m\"",
    "Content-Disposition: form-data; name=\"{N}",
    "Content-Disposition: form-data; name=\t\"{N}\"\t",
    "Content-Disposition: form-data; name=\"a\x01{N}\"",
    "Content-Disposition: form-data; name=\"{N}\"; filename*=UTF-8''x",
    "Content-Disposition:",
};

const std::vector<std::string> kOtherHeaders = {
    "X-Note: a: b", "NoColon", ": empty name",
    "Content-Disposition: form-data; name=\"second\""};

const std::vector<std::string> kContentTypes = {
    "Multipart/Form-Data; Boundary={B}",
    "multipart/form-data;boundary={B}",
    "  multipart/form-data  ;  boundary = {B}  ",
    "multipart/form-data; charset=utf-8; boundary={B}",
    "multipart/form-data; boundary={B};",
    "multipart/form-data; x; boundary={B}",
    "multipart/form-data; boundary={B}; boundary=C",
    "multipart/form-data; boundary=\"{B}\\\"\"",
    "multipart/form-data; boundary=\"{B}",
    "multipart/form-data; boundary=",
    "multipart/form-data",
    "multipart/form-dataX; boundary={B}",
    "multipart/mixed; boundary={B}",
    ""};

// A form with something clients do not send.
std::string OtherForm(std::mt19937_64& random, const std::string& boundary) {
  std::string body = Chance(random, 10) ? "preamble\r\n" : "";
  const int parts = std::uniform_int_distribution<int>(0, 4)(random);
  for (int part = 0; part < parts; ++part) {
    body += "--" + boundary + (Chance(random, 10) ? " \r\n" : "\r\n");
    if (Chance(random, 20)) {
      body += Pick(random, kOtherHeaders) + "\r\n";
    }
    body += Substitute(Chance(random, 50)
                           ? "Content-Disposition: form-data; name=\"{N}\""
                           : Pick(random, kDispositions),
                       "{N}", Pick(random, kNames)) +
            (Chance(random, 5) ? "\n\r\n" : "\r\n\r\n");
    body += Substitute(Pick(random, kContents), "{B}", boundary) + "\r\n";
  }
  return body + "--" + boundary +
         (Chance(random, 20) ? "--\r\nepilogue" : "--\r\n");
}

std::string Mutate(std::mt19937_64& random, std::string text) {
  static constexpr char kByteList[] =
      "-\r\n\"\\;=: \t\x0b"
      "aB\x00\x80\xff";
  static const std::string kBytes(kByteList, sizeof(kByteList) - 1);
  const int edits = std::uniform_int_distribution<int>(1, 3)(random);
  for (int edit = 0; edit < edits && !text.empty(); ++edit) {
    const std::size_t at =
        std::uniform_int_distribution<std::size_t>(0, text.size() - 1)(random);
    const char byte = kBytes[std::uniform_int_distribution<std::size_t>(
        0, kBytes.size() - 1)(random)];
    switch (std::uniform_int_distribution<int>(0, 3)(random)) {
      case 0:
        text.insert(text.begin() + static_cast<std::ptrdiff_t>(at), byte);
        break;
      case 1:
        text.erase(at, 1);
        break;
      case 2:
        text[at] = byte;
        break;
      default:
        text.resize(at);
        break;
    }
  }
  return text;
}

void TestRandomForms(Totals& totals) {
  std::mt19937_64 random(7578);
  for (int index = 0; index < 20000; ++index) {
    const std::string& boundary = Pick(random, kBoundaries);
    const std::string type =
        "multipart/form-data; boundary=" +
        (Chance(random, 20) ? "\"" + boundary + "\"" : boundary);
    const std::string body = ClientForm(random, boundary);
    Compare(totals, type, body);
    Compare(totals, type, OtherForm(random, boundary));
    Compare(totals, type, Mutate(random, body));
    if (Chance(random, 20)) {
      Compare(totals, Substitute(Pick(random, kContentTypes), "{B}", boundary),
              body);
    }
  }
  // Part headers around 16 KiB.
  for (const std::size_t padding : {16330U, 16340U, 16345U, 16350U}) {
    Compare(totals, "multipart/form-data; boundary=B",
            "--B\r\nX-Pad: " + std::string(padding, 'p') +
                "\r\nContent-Disposition: form-data; name=\"a\"\r\n\r\nv\r\n"
                "--B--");
  }
}

// Forms captured from clients: canonical, the same in every API, and read by
// the image API as their five fields.
bool TestClients(Totals& totals, std::size_t* clients) {
  bool ok = true;
  for (const auto& entry :
       std::filesystem::directory_iterator(GUFO_MULTIPART_FIXTURES)) {
    if (entry.path().extension() != ".form") {
      continue;
    }
    std::ifstream file(entry.path(), std::ios::binary);
    const std::string form{std::istreambuf_iterator<char>(file), {}};
    const std::size_t split = form.find("\r\n\r\n");
    constexpr std::string_view kHeader = "Content-Type: ";
    const std::string_view text(form);
    const std::string_view content_type =
        text.substr(kHeader.size(), split - kHeader.size());
    const std::string_view body = text.substr(split + 4);
    const std::size_t failures = totals.failures;
    Compare(totals, content_type, body);
    const Outcome image = New(Api::kImage, content_type, body, kNever);
    std::vector<std::string> names;
    for (const auto& field : image.fields) {
      names.push_back(field.name);
    }
    const bool read =
        Canonical(content_type, body) && totals.failures == failures &&
        image.status == Outcome::Status::kAccepted &&
        names == std::vector<std::string>{"model", "prompt", "file", "image[]",
                                          "image[]"} &&
        image.fields[1].content ==
            "Gr\xc3\xbc\xc3\x9f"
            "e \"quoted\"";
    if (!read) {
      std::fprintf(stderr, "%s is not read as captured\n",
                   entry.path().filename().string().c_str());
    }
    ok &= read;
    ++*clients;
  }
  return ok && *clients == 4;
}

// A field consumer's exception leaves ParseMultipart unchanged.
bool ExceptionsPropagate() {
  try {
    gufo::server::ParseMultipart(
        "multipart/form-data; boundary=B",
        "--B\r\nContent-Disposition: form-data; name=\"a\"\r\n\r\nv\r\n--B--",
        {.max_fields = 1, .unique_names = false},
        [](const MultipartField&) -> bool {
          throw std::invalid_argument("consumer");
        });
  } catch (const std::invalid_argument& error) {
    return std::string_view(error.what()) == "consumer";
  }
  return false;
}

}  // namespace

int main() {
  try {
    Totals totals;
    std::size_t clients = 0;
    const bool captured = TestClients(totals, &clients);
    TestRandomForms(totals);
    const bool exceptions = ExceptionsPropagate();
    if (!exceptions) {
      std::fprintf(stderr, "a field consumer's exception did not propagate\n");
    }
    // Seven messages and four other rejections.
    const bool covered = reached.size() == 11;
    if (!covered) {
      std::fprintf(stderr, "%zu of 11 rejections reached:\n", reached.size());
      for (const std::string& rejection : reached) {
        std::fprintf(stderr, "  %s\n", rejection.c_str());
      }
    }
    std::printf(
        "multipart forms: %zu canonical (%zu accepted reads), %zu failures; "
        "%zu clients; %zu others, %zu read differently\n",
        totals.canonical, totals.accepted, totals.failures, clients,
        totals.others, totals.others_differing);
    return totals.failures == 0 && captured && exceptions && covered ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
