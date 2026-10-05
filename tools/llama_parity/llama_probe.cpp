// For each tool schema of cases.py, print llama.cpp's parse of each candidate
// output and whether its own tool grammar (GBNF engine) admits it.
// Usage: llama_probe MODEL.gguf CASES.tsv (only the chat template is used).
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>

#include "../src/llama-grammar.h"
#include "../src/unicode.h"
#include "chat.h"
#include "llama.h"
int main(int argc, char** argv) {
  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.vocab_only = true;
  llama_model* model = llama_model_load_from_file(argv[1], mp);
  auto tmpls = common_chat_templates_init(model, "");
  std::ifstream in(argv[2]);
  std::string line;
  while (std::getline(in, line)) {
    // label \t schema \t candidate1 \t candidate2 ...
    std::vector<std::string> cols;
    size_t pos = 0, tab;
    while ((tab = line.find('\t', pos)) != std::string::npos) {
      cols.push_back(line.substr(pos, tab - pos));
      pos = tab + 1;
    }
    cols.push_back(line.substr(pos));
    common_chat_templates_inputs inputs;
    common_chat_msg u;
    u.role = "user";
    u.content = "call";
    inputs.messages = {u};
    common_chat_tool t;
    t.name = "record";
    t.description = "";
    t.parameters = cols[1];
    inputs.tools = {t};
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    inputs.enable_thinking = false;
    common_chat_params p;
    try {
      p = common_chat_templates_apply(tmpls.get(), inputs);
    } catch (const std::exception& e) {
      printf("%s\tAPPLY_ERROR %s\n", cols[0].c_str(), e.what());
      continue;
    }
    common_chat_parser_params pp(p);
    pp.parser.load(p.parser);
    auto grammar_accepts = [&](const std::string& text) {
      llama_grammar* g = llama_grammar_init_impl(
          nullptr, p.grammar.c_str(), "root", false, nullptr, 0, nullptr, 0);
      if (!g)
        return std::string("init-failed");
      for (uint32_t cpt : unicode_cpts_from_utf8(text)) {
        llama_grammar_accept(g, cpt);
        if (llama_grammar_get_stacks(g).empty()) {
          llama_grammar_free_impl(g);
          return std::string("no");
        }
      }
      bool done = false;
      for (const auto& st : llama_grammar_get_stacks(g))
        done |= st.empty();
      llama_grammar_free_impl(g);
      return std::string(done ? "yes" : "no");
    };
    for (size_t i = 2; i < cols.size(); ++i) {
      std::string text = nlohmann::json::parse(cols[i]).get<std::string>();
      try {
        // common_chat_parse adds pp.generation_prompt itself.
        auto m = common_chat_parse(text, false, pp);
        std::string args =
            m.tool_calls.empty() ? "-" : m.tool_calls[0].arguments;
        printf("%s\tcand%zu\tcalls=%zu\targs=%s\tgrammar=%s\n", cols[0].c_str(),
               i - 1, m.tool_calls.size(), args.c_str(),
               grammar_accepts(p.generation_prompt + text).c_str());
      } catch (const std::exception& e) {
        printf("%s\tcand%zu\tPARSE_ERROR\tgrammar=%s\n", cols[0].c_str(), i - 1,
               grammar_accepts(p.generation_prompt + text).c_str());
      }
    }
  }
}
