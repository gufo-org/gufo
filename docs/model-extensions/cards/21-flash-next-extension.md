# Card 21: Migrate Flash-Next text generation

Status: **pending**. RFC step and dependencies: **10 Flash-Next; 20**.

Future scope; do not execute during the cards 01–14 milestone.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

The throwaway Flash-Next spike informed this migration but did not establish production parity. Production factory/metadata selection, MTP ownership, decode concurrency identity and CLI drivers all require fresh qualifying evidence.

### Original scope

Move runner, MTP ownership/options, input handling, tool syntax, resolution, identity/decode-concurrency contribution and CLI drivers into extension. Affected actual modes/drafts, forced tool choice, cache/long-context, goldens and matched CLI/HTTP performance, default/lock artifacts, isolation ratchet.

## Implementation plan

1. Move Flash-Next runner, MTP heads/drafts/options, modality handling and tool syntax into its extension, using public family/runtime contracts.
2. Move artifact resolution, native/component continuation paths, exact identity lines including model-computed decode concurrency, and registered prompt/chat/bench drivers.
3. Wire real provider metadata/options and generated constants; remove remaining model branches from serving/CLI and ratchet isolation.
4. Reproduce actual off/MTP behavior and investigate any timing flags with fresh matched affected-history controls; do not reuse inconclusive spike results as a pass.

## Definition of done

- [ ] Production registration/parser/probe paths execute the correct modes and actual drafts with unchanged sampler/constraint semantics.
- [ ] Identity bytes, outputs, cache work and model-computed concurrency contributions remain exact.
- [ ] Independent numerical and matched CLI/HTTP quality/performance gates are complete for discovery and actual lock artifacts; spike limitations remain visible.

## Validation criteria

- Run Flash-Next QUALITY.md numerical checks and affected functional suites for every supported actual mode, including forced tool choice, long-context/cache and exact identity goldens.
- Run standard HTTP speed and matched CLI output/phase controls with retained binary/toolchain/model provenance; alternate only affected histories when investigating flags at unchanged shared thresholds.
- Run Linux/Windows builds, excluded-model/core-tool checks, HIP resource/production ISA audits and isolation ratchet. Verify production metadata selection rather than relying on the earlier synthetic spike probe.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
