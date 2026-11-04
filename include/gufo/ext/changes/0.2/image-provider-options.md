# Image provider startup options and request capabilities

Image (`src/ext/image.hpp`): add `ImageRunnerOptions`,
`ImageRunnerOptionsFactory` and `ImageProvider::create_with_options`, preferred
over `create`, so the selected provider receives its validated manifest options
for `serve image` (`kImageServeCommand`), for example a `--text-encoder` path.
Only explicitly supplied options of the provider that claims the model are
forwarded; another provider's options and missing `path` values are rejected
before the factory runs. Add `ImageBackground` (`kAuto`, `kOpaque`,
`kTransparent`) and `ImageRequest::background`: the HTTP background intent now
reaches the provider unchanged instead of being rewritten into prompt text.
Add `ImageCapabilities` (`edits`, `transparent_background`) with
`ImageRunner::Capabilities()` and `ImageRunner::ValidateRequest(...)`; the
image service rejects unsupported edits and transparent backgrounds, and
provider validation failures, with HTTP 400 before inference. Defaults
advertise neither feature; the Qwen-Image-2.1 runner advertises both and keeps
its RGBA prompt framing inside the provider. Behavior change for external
image runners: one that accepted reference images or transparent backgrounds
must now advertise them through `Capabilities()`. Options
(`include/gufo/ext/options.hpp`): add `OptionApplies`; a declaration for the
bare `serve` command applies to every `serve <modality>` command, as the Nix
serving helper already treats it, including for registry type-conflict checks.
