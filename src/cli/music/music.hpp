#ifndef GUFO_CLI_MUSIC_HPP_
#define GUFO_CLI_MUSIC_HPP_

#include <span>

namespace gufo::cli {

/// `gufo music`: one song from a music generation provider.
[[nodiscard]] int RunMusic(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_MUSIC_HPP_
