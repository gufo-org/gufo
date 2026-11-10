# JSON: write parsed numbers as read

JSON (`src/core/json.hpp`): a number read by `json::parse` keeps its validated
token, and serialisation can write that token instead of the `double`.
- New `json::Value::Numbers`: `kCanonical` (default) spells every number from
  its `double`, as before; `kAsParsed` writes the token `parse` read, so
  `9007199254740993`, `1.0`, `1e5` and `-0.0` round-trip unchanged.
- `Value::dump` and `Value::tojson` take an optional `Numbers`, which applies
  to nested values too. Numbers built from C++ values, or assigned over a
  parsed one, have no token and are always canonical.
- `as_double` and the other getters are unchanged; a token above 2^53 still
  reads as the nearest `double`.

Compatibility: additive. Calls without the new argument return the same text
as before. Code that takes the address of `dump` or `tojson` sees the new
parameter.
