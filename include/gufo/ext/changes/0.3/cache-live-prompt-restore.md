# Live slot prompt restore

Additive change to `cache::Adapter`. Providers can implement
`SaveLivePrompt` and `RestoreLivePrompt` for a checkpoint that only works in
the same execution slot. The provider charges its bounded storage as
per-session state and keeps any omitted append rows readable until the slot
is reset or reassigned. The host only restores a matching token prefix.
After a restore exception, the host invalidates the slot before reuse.

Migration is optional. Providers that leave the methods at their defaults
keep their existing checkpoint behavior.
