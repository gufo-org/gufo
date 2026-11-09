# Live prompt-admission headroom

`TextRunnerResourceClaim` adds the optional
`prompt_admission_available_bytes` callable. When provided alongside the
prompt-state byte rate and fixed budget, the server samples current host
headroom before reserving a prompt. It uses the smaller of that headroom and
the fixed budget, subtracting outstanding promises conservatively, including
reservations released during the sample. The callable may throw; admission
then fails without reserving prompt bytes. It must be thread safe: metrics may
sample it concurrently with admission. It must not re-enter the scheduler.

An empty callable preserves fixed-budget admission. Existing extensions remain
source compatible after rebuilding. The prompt-admission fields are appended
after the legacy fields, preserving positional aggregate initialization. This does not reserve generation growth
or guarantee memory availability after admission.
