# Card 28: Pin external compatibility fixtures and stability evidence

Status: **pending**. RFC step and dependencies: **12 preparation; 27**.

Future scope; do not execute during the cards 01–14 milestone.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Source compatibility requires pinned independent consumers. The pilot covers embeddings, so a separate external text fixture is required; updating those fixtures alongside the API would defeat the compatibility test.

### Original scope

Pin independent external EmbeddingGemma and text compatibility fixtures by repository/revision, never rewrite fixtures alongside API changes. Run every-release compatibility builds and scheduled conformance. Track measured SURFACE churn, bounded experimental text symbols (or zero experimental includes), every second provider and source-only stability policy/adapters. Publish dated evidence ledger.

## Implementation plan

1. Pin external EmbeddingGemma and text repositories/revisions with manifests, licenses and expected build/conformance commands in a dated compatibility ledger.
2. Wire every-release compatibility builds against those unchanged fixtures and scheduled real conformance runs; separate release build checks from weight-heavy scheduled jobs.
3. Measure SURFACE declaration/path churn and establish zero experimental includes for a text consumer or a bounded published enforced symbol list.
4. Track second-provider coverage, source-only compatibility/adapters and dated pilot CMake/Nix/text-header stability evidence required by Card 29; do not shorten the window.

## Definition of done

- [ ] Both independent embedding and text fixtures are pinned and not rewritten alongside API changes.
- [ ] Release builds and scheduled conformance execute against those pins with retained results.
- [ ] Measured surface churn, experimental symbol policy, second providers and dated stability ledger are sufficient to audit 1.0 eligibility without asserting elapsed time prematurely.

## Validation criteria

- Build the pinned external fixtures from clean directories against candidate/release Gufo using their recorded commands; verify source hashes/revisions remain unchanged and compile only public permitted headers.
- Run scheduled capability conformance and inspect actual reports/CI invocations; a configured workflow without a successful run is incomplete evidence.
- Run SURFACE/API/isolation gates, audit bounded experimental symbols and ledger dates, and preserve any compatibility break with the appropriate version/adapters rather than patching the fixture.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
