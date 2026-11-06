# Shared provider options

Public metadata contract: a shipped and an unshipped extension may declare the same
option when the type agrees, as extensions in one tier already could. This lets an
external provider declare a shared flag such as `--mtp-model` and map it from
`serving.draft_options`. Disagreeing types still fail. Artifact keys and model IDs
still may not collide across tiers. Additive.
