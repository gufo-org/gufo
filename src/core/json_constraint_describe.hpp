#ifndef GUFO_CORE_JSON_CONSTRAINT_DESCRIBE_HPP_
#define GUFO_CORE_JSON_CONSTRAINT_DESCRIBE_HPP_

#include <string>

#include "src/core/json_constraint.hpp"

namespace gufo::sampling {

// A deterministic text of everything a compiled grammar holds: rules, byte
// classes, argument sets, roots, prompt and flags, and each lexeme's
// alphabet and answers on fixed probes. Two grammars with the same text
// constrain output the same way; tests use it to prove a compiler change
// leaves grammars unchanged.
std::string DescribeJsonConstraint(const JsonConstraint& grammar);

}  // namespace gufo::sampling

#endif
