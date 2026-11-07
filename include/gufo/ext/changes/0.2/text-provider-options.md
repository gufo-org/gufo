# Text provider options

Text (`src/ext/text.hpp`, `include/gufo/ext/options.hpp`): the options a text
provider declares in its manifest reach its runner and drivers.
- `TextRunnerOptions::provider_options`, `TextPromptOptions::provider_options`
  and `TextBenchOptions::provider_options` (`ParsedOptions`): the explicit
  command-line values of the options the selected provider declares for
  `serve`, `prompt`/`chat` or `bench`, keyed by long name. An option that was
  not given is absent; explicit `false` and `0` are present.
- `ext::FindOption<T>(options, name)`: a pointer to the value, null when absent;
  `T` follows the declared type (`bool`, `std::int64_t`, `double`,
  `std::string`).
- The commands accept every option an available text provider declares for
  them (other capabilities' options stay with their own commands) and list them under "Provider options" in `--help`. Options of other providers or
  commands, type errors and repeated options fail before weights load.
- Declared spellings core parses itself (`--mtp-model`, ...) keep their core
  handling and fields and are not delivered in `provider_options`.

Compatibility: additive. Existing providers receive an empty payload unless they
declare options, and command lines accepted before are accepted unchanged.
