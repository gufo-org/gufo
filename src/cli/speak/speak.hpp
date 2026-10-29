#ifndef GUFO_CLI_SPEAK_HPP_
#define GUFO_CLI_SPEAK_HPP_

#include <span>
#include <string_view>

namespace gufo::cli {

void PrintSpeakHelp(std::string_view program_name);
[[nodiscard]] int RunSpeak(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_SPEAK_HPP_
