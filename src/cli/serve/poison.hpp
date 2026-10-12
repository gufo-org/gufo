#ifndef GUFO_CLI_SERVE_POISON_HPP_
#define GUFO_CLI_SERVE_POISON_HPP_

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

namespace gufo::cli {

/// A device fault leaves the HIP context unusable: every later request then
/// fails in milliseconds with a bare "generation failed" and nothing useful in
/// the log. On the first such fault, log the original detail once and leave
/// with a non-zero status rather than serving those failures. Returns normally
/// when the error is not a device fault, or when GUFO_KEEP_POISONED=1 asks for
/// the old behaviour.
///
/// Inline so the header can be included from any translation unit.
inline void ExitOnPoisonedContext(std::string_view detail) {
  // Markers the driver and the runtime use for a context that can no longer run
  // work, as opposed to a request-level failure.
  static constexpr std::array<std::string_view, 5> kDeviceFaults{
      "illegal memory access", "MEMORY_APERTURE_VIOLATION", "HSA_STATUS",
      "device fault", "memory fault"};
  bool fatal = false;
  for (const auto marker : kDeviceFaults) {
    fatal = fatal || detail.find(marker) != std::string_view::npos;
  }
  if (!fatal) {
    return;
  }
  if (const char* keep = std::getenv("GUFO_KEEP_POISONED");
      keep != nullptr && keep[0] != '\0' && keep[0] != '0') {
    return;
  }
  // Written straight to stderr, not through Logger, so the header stays
  // independent of the logging library.
  std::fprintf(stderr,
               "fatal: device fault, the GPU context is unusable; exiting "
               "non-zero instead of serving failed requests\n  detail: %.*s\n",
               static_cast<int>(detail.size()), detail.data());
  std::fflush(nullptr);
  std::_Exit(70);
}

}  // namespace gufo::cli

#endif  // GUFO_CLI_SERVE_POISON_HPP_
