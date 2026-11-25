#pragma once

#include <gufo/ext/text.hpp>
#include <memory>
#include <string>

namespace gufo::models::inkling2 {

/// Runner factory: loads the engine, the tokenizer and (optionally) the vision
/// encoder. Returns nullptr with `*error` set when the model cannot be served
/// (for example when speculative decoding is requested, which this extension
/// does not implement).
[[nodiscard]] std::shared_ptr<ext::TextModelRunner> CreateInkling2TextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

}  // namespace gufo::models::inkling2
