# JSON parsing out of line

`gufo::json::parse` (`src/core/json.hpp`, reached through `gufo/ext/json.hpp`)
is no longer an inline function; Gufo defines it. It keeps its declaration,
the text it accepts, the values and number tokens it returns and its error
messages. Besides `std::runtime_error` on malformed input and
`std::bad_alloc`, it may now throw `std::logic_error` on a parser bug.

Compatibility: source compatible after rebuilding. Extension modules resolve
`parse` from the Gufo executable; executables link it through
`gufo_ext_runtime` or `gufo_ext_core`.
