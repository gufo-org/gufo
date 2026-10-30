#ifndef GUFO_SERVER_TRACE_HPP_
#define GUFO_SERVER_TRACE_HPP_

#include <optional>
#include <string>
#include <string_view>

#include "src/core/json.hpp"

namespace gufo::server {

/// Opt-in client-content trace written by `gufo serve llm --trace PATH`.
///
/// Logs never carry prompt text, message bodies or generated text. This sink
/// does, so a client-visible failure (a leaked tool-call closer, a history the
/// template renders differently) can be read from what the server actually
/// received, rendered, generated and sent. Each record is one JSON object per
/// line carrying `time`, `event` and the `request` id that the matching
/// `[http] request=` log lines name. Nothing is written unless an operator
/// names a file.
class Trace {
public:
  /// Opens `path` for appending, creating it readable by the owner only. Call
  /// once during startup, before any request thread starts. Returns the reason
  /// on failure so the CLI can report it.
  static std::optional<std::string> Open(const std::string& path);

  /// Closes the sink; later records are dropped.
  static void Close();

  /// False when `Write()` would drop the record. Call sites check this before
  /// building one: decoding a whole prompt is not free.
  [[nodiscard]] static bool Enabled();

  /// A record object starting with `time`, `event` and, when not empty,
  /// `request`. Callers add their fields and pass it to `Write()`.
  [[nodiscard]] static json::Value Record(std::string_view event,
                                          std::string_view request_id);

  /// Appends `record` as one line. Content strings go through `Text()` so
  /// every line stays valid JSON. A failure drops the record, warns once and
  /// never reaches the caller; a partly written record is removed, and a sink
  /// that cannot remove it is closed.
  static void Write(const json::Value& record);

  /// `bytes` with malformed UTF-8 replaced by U+FFFD.
  [[nodiscard]] static std::string Text(std::string_view bytes);

  /// Binds an HTTP request id to the calling thread for the scope's lifetime,
  /// so the text scheduler can key its record without the id travelling
  /// through every backend signature. Backends submit generation work on the
  /// connection thread that runs the handler, inside this scope.
  class RequestScope {
  public:
    explicit RequestScope(std::string_view request_id);
    ~RequestScope();

    RequestScope(const RequestScope&) = delete;
    RequestScope& operator=(const RequestScope&) = delete;
    RequestScope(RequestScope&&) = delete;
    RequestScope& operator=(RequestScope&&) = delete;

  private:
    std::string previous_;
  };

  /// The id bound by the innermost `RequestScope` on this thread, or empty.
  [[nodiscard]] static std::string CurrentRequest();
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_TRACE_HPP_
