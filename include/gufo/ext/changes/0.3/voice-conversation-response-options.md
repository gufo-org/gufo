# Voice conversation response options

Voice conversations (`src/ext/asr.hpp`):
- `VoiceConversation::Settings::response_instructions` carries a reply's own
  `response.instructions`; Respond gives them after the conversation as a
  system message.
- `VoiceConversation::Settings::call_tools` is false for `tool_choice: "none"`
  (session or reply) and for a reply's `tools: []`: the reply must not call
  the tools the context lists.
- `VoiceConversation::Settings::out_of_band` marks a reply with
  `response.conversation: "none"`: it leaves the context when it ends.
- `VoiceConversation::supports_response_options()` reports whether a
  conversation honours these. The default is false, and clients asking for
  any of them get an error event.

Gufo folds `role: "system"` message items into `Settings::instructions`
after the session's instructions, so conversations need no new turn kind.
Existing conversations keep compiling and behave as before. They must be
rebuilt against the new header.
