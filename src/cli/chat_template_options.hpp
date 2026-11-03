#ifndef GUFO_CLI_CHAT_TEMPLATE_OPTIONS_HPP_
#define GUFO_CLI_CHAT_TEMPLATE_OPTIONS_HPP_

#include "src/cli/arg_parser.hpp"
#include "src/core/jinja/chat_template.hpp"

namespace gufo::cli {
inline void RegisterChatTemplateOptions(ArgParser& parser,
                                        core::ChatTemplateConfig& config) {
  parser.AddFlag("", "--chat-template",
                 "Render the Jinja chat template embedded in the GGUF",
                 "Prompt", &config.enabled);
  parser.AddCustomOption(
      "", "--chat-template-path", "PATH",
      "Override the GGUF template with a file (requires --chat-template)",
      "Prompt",
      [&config](std::string_view, std::string_view path, std::string* error) {
        if (path.empty()) {
          if (error)
            *error = "--chat-template-path requires a nonempty file path";
          return false;
        }
        config.path = path;
        return true;
      });
}
}  // namespace gufo::cli
#endif
