# External chat renderers

Registration-only extensions may use `providers: []` and register a
`ChatRendererFactory`. HTTP model loading selects at most one renderer by provider
ID and passes it through `TextRunnerOptions::chat_renderer` for load-time
validation ownership. The host installs it once with
`TextModelRunner::SetChatRenderer`. Shared `PrepareChatPrompt`, `CountChatTokens`,
`RenderChatPrompt` and `ChatInitialOutputState` handle external rendering and
fall back to the existing native methods when no renderer is installed.

`TextModelRunner::PrepareRenderedPrompt` is the model-owned input primitive:
it tokenizes validated framing and prepares supported image context. All shipped
text providers implement it. The default rejects unsupported preformatted input;
existing native runner implementations remain source-compatible.
`TextPreparedPrompt` is unchanged.

`ChatMessage::content_parts` preserves validated text/image parts without URLs.
Developer image parts reach the model so a formatter can support image examples.
`ChatRequest::chat_template_kwargs` retains custom Chat Completions and Responses
arguments; resolved reasoning controls remain authoritative. `RenderedChatPrompt`
supplies native text, ordered image-placeholder offsets, literal byte ranges and
the initial output state. The model retains tokenization and vision preprocessing.

This is additive. The extension owns template loading, execution and configuration;
no interpreter, template language, CLI switch or binary plugin ABI is introduced.
