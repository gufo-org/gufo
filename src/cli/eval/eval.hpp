#ifndef GUFO_CLI_EVAL_EVAL_HPP_
#define GUFO_CLI_EVAL_EVAL_HPP_

#include <span>
#include <string_view>

namespace gufo::cli {

void PrintEvalHelp(std::string_view program_name);
int RunEval(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_EVAL_EVAL_HPP_
