# Card 13: Migrate the EmbeddingGemma pilot

Status: **pending**. RFC step and dependencies: **8 embeddings; 12**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

EmbeddingGemma is the first real provider and the prerequisite for metadata-generated infrastructure. It already owns batch scheduling and audio decoding; introducing a new core admission queue would change the pilot scope.

### Original scope

Migrate EmbeddingGemma pilot, extension-relative includes, metadata/options/pure probe/artifact resolution/provider/model IDs. Preserve extension batch scheduling and audio decoder ownership. Move model production dependency/data/runtime-library declarations into manifests. Existing API/numerical quality/speed gates, default and lock artifact, isolated build and fail-on-violation ratchet.

## Implementation plan

1. Move provider implementation, service-facing model calls and extension-relative own includes under the EmbeddingGemma extension boundary using Card 12 test support.
2. Declare metadata, canonical options, HIP-free pure artifact probe, artifact resolution, provider/model IDs and factory registration with generated constants.
3. Move model production dependency/data/runtime-library declarations into extension manifests, preserving libav/audio decoder ownership and current batch scheduler behavior.
4. Remove migrated pilot knowledge from core routing/build paths, populate selection/lock as an explicit release decision, and turn its isolation boundary into fail-on-violation enforcement.

## Definition of done

- [ ] Pilot loads through the registry with truthful probe/selection/errors and unchanged advertised IDs, modalities, batch ordering and scheduling.
- [ ] Its manifest owns production packages, data and runtime libraries; isolated/excluded builds work without hidden root declarations.
- [ ] Existing API, independent numerical and speed gates pass on both discovery and actual lock-mode HIP artifacts; ratchet rejects new coupling.

## Validation criteria

- Run the full embedding suite via tests/functional/model_services.py embedding --embedding-suite full with the production binary/model and a new output directory, as documented in tests/functional/README.md.
- Run existing EmbeddingGemma QUALITY.md numerical commands and standard speed controls; full service observations alone do not qualify matched performance. Retain independent reference vectors and per-request work/timing evidence.
- Build/test the pilot alone with required family dependencies and with it excluded, on Linux/Windows. Inspect installed data/runtime libraries and run the fail-on-violation isolation check; repeat affected runtime gates on the actual lock artifact.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
