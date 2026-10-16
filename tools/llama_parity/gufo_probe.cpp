// Print whether gufo's native tool grammar admits each candidate of cases.py.
// Usage: gufo_probe CASES.tsv qwen|ds auto|required
#include <fstream>
#include <iostream>

#include "src/core/json_constraint.hpp"
using namespace gufo::sampling;
bool Accepts(const JsonConstraint& g, std::string_view t) {
  auto s = g.Start();
  for (unsigned char b : t) {
    s = g.Advance(s, b);
    if (s.empty())
      return false;
  }
  return g.Complete(s);
}
int main(int, char** argv) {
  std::ifstream in(argv[1]);
  std::string line;
  while (std::getline(in, line)) {
    std::vector<std::string> cols;
    size_t pos = 0, tab;
    while ((tab = line.find('\t', pos)) != std::string::npos) {
      cols.push_back(line.substr(pos, tab - pos));
      pos = tab + 1;
    }
    cols.push_back(line.substr(pos));
    std::shared_ptr<const JsonConstraint> p;
    try {
      p = JsonConstraint::ToolParameters(
          gufo::json::parse(cols[1]), false,
          (std::string(argv[2]) == "ds" ? JsonConstraint::ToolFormat::kDeepSeek
                                        : JsonConstraint::ToolFormat::kQwen));
    } catch (const std::exception& e) {
      std::cout << cols[0] << "\tERROR " << e.what() << "\n";
      continue;
    }
    if (!p) {
      std::cout << cols[0] << "\tJSON_FALLBACK\n";
      continue;
    }
    auto g = JsonConstraint::WithTools(
        nullptr, {{"record", p}}, std::string(argv[3]) == "required", false,
        (std::string(argv[2]) == "ds" ? JsonConstraint::ToolFormat::kDeepSeek
                                      : JsonConstraint::ToolFormat::kQwen));
    for (size_t i = 2; i < cols.size(); ++i)
      std::cout << cols[0] << "\tcand" << i - 1 << "\t"
                << (Accepts(*g, gufo::json::parse(cols[i]).str()) ? "yes"
                                                                  : "no")
                << "\n";
  }
}
