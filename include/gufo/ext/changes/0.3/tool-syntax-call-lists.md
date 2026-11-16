# Tool syntaxes for call lists, keyword arguments and repeated names

`JsonConstraint::ToolSyntax` can now describe two more native call spellings.

- `block_close` (with or without tagged arguments) makes every call share one block: `open` once, calls joined
  by the new `call_separator`, then `block_close`. This is how a Python-style call list
  (`[NAME(k='v'), OTHER()]`) is written. `call_separator` also separates consecutive calls that each repeat `open`.
- `ValueSyntax::keyword_root` writes the root parameter object as keyword arguments (`k=v, n=3`, no braces; the
  call's `name_suffix` and `close` supply the parentheses). Root strings are JSON strings or `string_open`/
  `string_close` strings, root booleans and null JSON's or Python's; everything nested keeps JSON's spelling.
  Property names must be identifiers.
- `repeat_prefix` / `repeat_suffix` make a call write its name a second time after `name_suffix`
  (`to=NAME<|message|><invoke name="NAME">`).

The change is additive: every new field defaults to empty/false, which keeps the previous grammars byte for byte.
Migration: none.
