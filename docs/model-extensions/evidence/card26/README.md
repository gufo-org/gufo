# Card 26 distribution evidence

Card 26 remains **pending**. These results cover repository implementation and
Linux builds; published-container execution and native Windows validation remain
open. No model quality qualification is claimed.

## Implementation

The [author guide](../../../extensions.md) uses the independent LFM3 extension
at `fb488aebef5abc3fefcbb21ce0d53eebe8cc3e27` as an architectural example.
[Distribution policy](../../DISTRIBUTION.md) defines core, community, promotion
and abandonment: niche/community extensions stay out of tree until promoted.
[Conformance guidance](../../CONFORMANCE.md) covers every capability, provenance
and the distinction between contract tests and inference qualification.

## Executed checks

| Check | Result | Retained output |
| --- | --- | --- |
| Distribution audit helpers | 7 passed | [output](distribution-tests.txt) |
| Extension discovery, including installed template license | 19 passed | [output](discovery-tests.txt) |
| Metadata configuration / schema | 14 / 2 passed | [configuration](metadata-tests.txt), [schema](schema-tests.txt) |
| API compatibility policy | 19 passed | [output](api-policy-tests.txt) |
| Capability contracts | 7 passed | [output](capability-contracts.txt), [build](contract-build.txt) |
| Windows bundle unit tests on Linux | 7 passed | [output](windows-bundle-tests.txt) |
| Dependency inventory | Passed, 39 components | [output](dependencies.txt), [inventory](dependency-inventory.json) |

The capability selection was `text_model_runner_test`,
`text_generation_scheduler_test`, `embedding_gemma2_embeddings_api_test`,
`qwen3_asr_audio_api_test`, `qwen3_tts_audio_api_test`, `qwen_image_21_test`, and
`video_api_test`, run using `ctest --preset gpu-full -R` after building those
specific targets. These weight-free contracts do not establish model quality.
Initial failed metadata and Windows test-environment runs are retained alongside
the successful runs; the Windows tests use mocked DLL inputs, not Windows.

## Installed packages

A clean source snapshot was configured with HIP, tools and tests disabled,
`GUFO_EXTENSIONS=template_model`, and an explicit copied external template.
[Snapshot identity](source-snapshot.json), [configuration](source-configure.txt),
[build](source-build.txt) and [installation](source-install.txt) are retained.
The [installed audit](source-install-audit.json) passed binary/provenance,
license and data checks. [Excluded assets](excluded-assets.json) confirms that
unselected H3 and DeepSeek assets were not installed.

The pinned template was built with
`nix build --offline path:/tmp/gufo-card26-template --no-link --max-jobs 1 --cores 4 --print-out-paths`.
Its Gufo input is the published Card 25 revision
`ae7ef25f350650a94be2f798706895fa0c8aa80f`, not the Card 26 candidate.
[Build output](template-nix-build.txt) and [audit](template-nix-audit.json) retain
the actual package and installed-file hashes. The copied template predates final
documentation link edits; its build inputs are unchanged.

The candidate release was built with
`nix build --offline .#release --no-link --max-jobs 1 --cores 4 --print-out-paths`.
[Build output](release-build.txt) and [audit](release-nix-audit.json) verify lock
mode, extension provenance, installed licenses, H3 data and the wrapped ELF.
The audit hashes explicit installed files; it is not a full runtime-library or
inference qualification. [Windows dependency inventory](windows-dependencies.json)
passed the lock-selected manifest check; native Windows packaging is unexecuted.

## External publication and open gates

Toolboxes source revision `62d93f839ac3341e5228dbe64e14c4a6ddd27cca`
pins Gufo `2e6aa96ce20a0e244561cb86915dd79260042eac`.
[Source inspection](toolboxes-source.json), [published image manifests](published-images.json)
and [build excerpts](toolboxes-build-excerpts.txt) retain the observed identifiers.
The [published build](https://github.com/gufo-org/toolboxes/actions/runs/41301781526)
succeeded; anonymous registry manifest/config inspection verified the digests.

Both published images were subsequently executed in the local Docker daemon;
see the [local verification and findings](toolboxes-local-check.md). Runtime GPU
diagnostics passed, but the development image failed Gufo source configuration.
A successful extension build inside the published builder and lock-mode
runtime/provenance checks remain required.
The inspected edge configuration does not explicitly enable extension lock mode.
Native Windows installation, runtime DLL closure and smoke checks also remain
required. Repository recipes and image publication alone do not close this card.
