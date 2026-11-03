# Optional Jinja chat templates

Add `gufo/ext/chat_template.hpp` with immutable native Jinja rendering, source
selection and input-provenance parts. `TextRunnerOptions` and
`TextPromptOptions` can carry the initialized renderer; prompt options also
carry the CLI source selection. Providers retain their compiled formatters when
no renderer is supplied.

Add `ChatMessage::template_content` for validated content-part arrays,
`ChatRequest::chat_template_kwargs` for request-specific template inputs, and
`TextRunnerCapabilities::uses_jinja_chat_template` to preserve those parts only
for a provider using Jinja. Custom templates retain the provider's native
reasoning and tool-call response conventions. Renderer identities include the
interpreter version and template content for persistent cache isolation.
