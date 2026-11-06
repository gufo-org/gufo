#ifndef GUFO_CLI_DECIDE_HPP_
#define GUFO_CLI_DECIDE_HPP_
#include <span>
namespace gufo::cli {
int RunDecide(std::span<const char* const> args);
}
#endif
