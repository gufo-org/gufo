# Image provider startup options and request capabilities

Image (`src/ext/image.hpp`): add `ImageRunnerOptions`,
`ImageRunnerOptionsFactory` and `ImageProvider::create_with_options`, preferred
over `create`, so the selected provider receives its validated manifest options
for `serve image` (`kImageServeCommand`), for example a `--text-encoder` path.
Only explicitly supplied options of the provider that claims the model are
forwarded; another provider's options and missing `path` values are rejected
before the factory runs. Add `ImageBackground` (`kAuto`, `kOpaque`,
`kTransparent`) and `ImageRequest::background`.

Add `ImageCapabilities` (`edits`, `transparent_background`) with optional
`ImageRunner::Capabilities()` and `ImageRunner::ValidateRequest(...)`.
An explicit capability declaration opts into unchanged prompts and
provider-owned background conditioning. The image service rejects unsupported
edits, transparent backgrounds and provider validation failures with HTTP 400
before inference. An explicit empty declaration accepts neither feature.
The default nullopt declaration preserves legacy edit acceptance and host RGBA
prompt framing for transparent requests. Qwen-Image-2.1 explicitly advertises
both features and keeps its RGBA prompt framing inside the provider.
These additions preserve existing runners after rebuilding.

Options (`include/gufo/ext/options.hpp`): add `OptionApplies`; a declaration for
the bare `serve` command applies to every `serve <modality>` command, as the
Nix serving helper already treats it, including for registry type-conflict
checks and shared CLI option registration.
