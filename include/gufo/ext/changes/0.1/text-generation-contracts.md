# Text generation contracts

Publish capability-neutral text generation contracts and registries:
- Runner descriptors declare their sampling preset so omitted HTTP settings
  retain model-specific defaults after extension loading.
- Text (`include/gufo/ext/text.hpp`, `src/ext/text.hpp`):
  `TextRunnerToken`, `TextPromptContext`, `TextPreparedPrompt`,
  `TextRunnerRamCacheOptions`, `TextRunnerDiskCacheOptions`,
  `TextExecutionPlanKind`, `TextExecutionPlan`, `TextRunnerCapabilities`,
  `TextRunnerPersistenceDescriptor`, `TextRunnerDescriptor`,
  `TextRunnerResourceClaim`, `TextRunnerMeasuredResources`,
  `TextPrefillStep`, `TextRunnerNgramRows`, `TextDecodeSelection`,
  `TextDecodeStep`, `TextRunnerState`, `TextRunnerSnapshot`,
  `TextRunnerAdvance`, `TextRunnerDecode`, `InitialOutputState`,
  `ChatRole`, `ChatMessage`, `ChatTool`, `ChatRequest`,
  `TextModelRunner`, `TextRunner`, `HipTextModelRunner`,
  `TextRunnerOptions`, `TextProvider`,
  `RegisterTextProvider`, `FindTextProvider`, `ProbeTextProvider`,
  `Registry::AddTextProvider`.
- Tool Dialect (`src/ext/text.hpp`):
  `ToolDialect`, `RegisterToolDialect`, `FindToolDialect`, `AllToolDialects`.
- Extension migrations for Cards 20–22: Qwen 27B (`qwen`),
  Qwen3.8 Flash-Next (`qwen38_flash_next`), and DeepSeek V4 Flash (`deepseek_v4_flash`).
