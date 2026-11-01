#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/models/deepseek_v4_flash/probe.hpp"
#include "src/models/deepseek_v4_flash/tool_calls.hpp"

namespace gufo::models::deepseek_v4_flash {

void RegisterToolDialect() {
  ext::RegisterToolDialect({
      .format = sampling::JsonConstraint::ToolFormat::kDeepSeek,
      .markers =
          {
              DeepSeekToolSyntax::kCallsOpenWithSeparator,
              DeepSeekToolSyntax::kCallsOpen,
          },
      .reasoning_markers =
          {
              DeepSeekToolSyntax::kNativeCallOpenWithSeparator,
              DeepSeekToolSyntax::kNativeCallOpen,
          },
      .closers =
          {
              DeepSeekToolSyntax::kParameterCloses[0],
              DeepSeekToolSyntax::kInvokeCloses[0],
              DeepSeekToolSyntax::kCallsClose,
          },
      .scan_tool_calls = &ScanDeepSeekToolCalls,
  });
}

}  // namespace gufo::models::deepseek_v4_flash

void deepseek_v4_flash_register(gufo::ext::Registry&) {
  gufo::models::deepseek_v4_flash::RegisterToolDialect();
}
