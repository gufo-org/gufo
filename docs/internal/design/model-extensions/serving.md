# Nix serving from extension metadata

`lib.mkGufoServe` (also available as `lib.mkServe`) renders a shell command.
Model-specific option arguments come from the selected package's extension
manifests. For example, `--mtp-model` becomes `mtpModel`; path, string, integer,
number and boolean options retain their manifest types. Shared declarations must
agree on their type. Unknown arguments and incompatible provider combinations
fail during Nix evaluation.

```nix
let
  package = gufo.packages.x86_64-linux.default;
in gufo.lib.mkGufoServe {
  gufo = package;
  model = "/var/lib/gufo/flash-next.gguf";
  provider = "flash-next-q4";
  speculative = "mtp";
  mtpModel = "/var/lib/gufo/mtp.gguf";
  port = 8080;
}
```

`provider` selects metadata for validation. It does not override runtime artifact
probing or emit a CLI flag. Without it, the configuration must be supported by
at least one built provider of the requested modality. The CLI still validates
the actual loaded model. Evaluation does not open model or draft files on the
deployment host; model and required draft paths must be nonempty.

## Canonical arguments

The existing names remain canonical: `dflashModel` / `--dflash-model`,
`dsparkModel` / `--dspark-model`, and `mtpModel` / `--mtp-model`. They identify
separate provider-owned options, with no aliases or public argument rename.
`speculative` values come from providers, plus the host's `off` value.
The host maps capabilities to serving commands; available modalities come from
the selected providers. Host network, service-user and restart policy stay with
the caller.

## Manifest declarations

A provider can declare its required draft option for each speculative mode:

```json
{
  "speculative": ["mtp"],
  "serving": {"draft_options": {"mtp": "--mtp-model"}}
}
```

The referenced option must be declared as a path for that provider and the
`serve` command. CMake validates these references. A speech-synthesis provider
advertises `"serving": {"voices": true}` to accept named WAV references through
`voices`. Each voice is a WAV path or `{ wav = ...; text = ...; language = ...; }`.
Voice names, paths, transcripts and language values remain deployment inputs.

## Package selection and external sources

The default package uses discovery; the release package uses `extensions.lock`
and release version identity. `mkGufoServe` consumes the chosen package's
`extensionMetadata` passthru, including its source root and experimental
`extensions` override. External roots follow CMake discovery: either a root
containing `extension.json` or a directory of extension roots. An external
extension absent from the lock is unavailable in a release configuration.

```nix
let
  package = gufo.packages.x86_64-linux.default.override {
    extensions = [ /path/to/my-extension ];
  };
in gufo.lib.mkGufoServe {
  gufo = package;
  modality = "llm";
  model = "/var/lib/gufo/model.gguf";
  provider = "my-provider";
}
```

A string/path executable override has no package metadata; the helper falls back
to the repository's discovery manifests. Use a package override to validate a
custom extension selection. `extraArgs` remains an explicit caller-managed
escape hatch and is not checked against manifests.

The current `serve llm` parser still explicitly registers its existing draft
flags and speculative modes. Rendering a newly declared option or speculative
mode in Nix does not by itself implement its CLI parsing or execution. External
providers using the existing serving contract work without such additions;
new runtime controls need serving integration as well as metadata.

## Checks

```sh
nix build .#checks.x86_64-linux.mk-serve
nix build .#checks.x86_64-linux.mk-serve-nixos
```

The first check covers CLI rendering and positive/negative metadata evaluation
fixtures, including an external template. The second evaluates real NixOS
systemd service definitions with both default and release packages. It checks
command rendering and artifact selection; it does not boot a VM or establish
GPU/model readiness.
