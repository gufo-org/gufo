#ifndef GUFO_SERVER_LOGGING_HPP_
#define GUFO_SERVER_LOGGING_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace gufo::server {

/// Log verbosity, ordered from chattiest to quietest so a threshold test is a
/// single comparison. `kDebug` must stay below `kInfo`: the process-wide filter
/// emits a line when `level >= Level()`.
enum class LogLevel : std::uint8_t {
  kDebug,
  kInfo,
  kWarn,
  kError,
};

/// Parses `error`, `warn`, `info` or `debug`. Returns nullopt for anything else
/// so a CLI can report its own message.
std::optional<LogLevel> LogLevelFromName(std::string_view name);

/// The spelling `LogLevelFromName` accepts, for echoing the resolved level
/// back.
std::string_view LogLevelName(LogLevel level);

class Logger {
public:
  /// Sets the process-wide verbosity threshold. Call once during startup,
  /// before any loader, scheduler or HTTP work starts: logging is shared by
  /// every modality, so a per-backend copy would only create a second source
  /// of truth. Default `kInfo`.
  static void SetLevel(LogLevel level);

  /// The current threshold. Prefer `Enabled()` at call sites that build a
  /// message before logging it.
  [[nodiscard]] static LogLevel Level();

  /// False when `Log(level)` would discard the line. Expensive lines should
  /// check this before formatting: `MemoryStatus()` reads /proc twice.
  [[nodiscard]] static bool Enabled(LogLevel level);

  /// Logs a general server or engine message:
  /// YYYY-MM-DD HH:MM:SS [LEVEL] [component] message
  static void Log(LogLevel level, std::string_view component,
                  std::string_view message);

  /// Logs a printf-formatted message through `Log()`, dropping it before any
  /// formatting when the threshold filters the level. The imported C-style
  /// runtimes (DeepSeek V4 Flash) use this instead of writing to stderr
  /// directly, so their informational lines obey the same process-wide level.
  static void LogFormatted(LogLevel level, std::string_view component,
                           const char* format, ...)
#if defined(__GNUC__) || defined(__clang__)
      __attribute__((format(printf, 3, 4)))
#endif
      ;

  /// Logs completion after the response body, including streamed generation.
  /// `success_level` is used when the status and outcome are unremarkable, so
  /// endpoints that are quiet by default can appear only under `-v` while
  /// 4xx/5xx and disconnects keep escalating to `kWarn`/`kError`.
  static void LogRequest(std::string_view id, std::string_view method,
                         std::string_view path, int status_code,
                         double duration_ms, std::string_view details = "",
                         std::string_view outcome = "completed",
                         LogLevel success_level = LogLevel::kInfo);

  /// Process RSS and host available memory; these are not additive GPU totals.
  static std::string MemoryStatus();

  /// Convenience helpers
  static void Debug(std::string_view component, std::string_view message) {
    Log(LogLevel::kDebug, component, message);
  }

  static void Info(std::string_view component, std::string_view message) {
    Log(LogLevel::kInfo, component, message);
  }

  static void Warn(std::string_view component, std::string_view message) {
    Log(LogLevel::kWarn, component, message);
  }

  static void Error(std::string_view component, std::string_view message) {
    Log(LogLevel::kError, component, message);
  }
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_LOGGING_HPP_
