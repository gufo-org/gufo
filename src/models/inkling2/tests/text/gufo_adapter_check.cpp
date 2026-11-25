// Compile/link check for gufo_adapter.hpp against a Gufo source tree:
//   clang++ -std=c++20 -fsyntax-only -I$GUFO -I$GUFO/include -I../../src
//   gufo_adapter_check.cpp
#include "src/models/inkling2/text/gufo_adapter.hpp"

int main() {
  gufo::ext::ChatRequest req;
  req.messages.emplace_back(gufo::ext::ChatRole::kUser, "hi");
  (void)inkling2::text::gufo_adapter::ToMessages(req);
  (void)inkling2::text::gufo_adapter::BudgetTags();
  return 0;
}
