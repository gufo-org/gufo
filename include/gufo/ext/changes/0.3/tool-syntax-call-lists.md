# Call lists, keyword arguments and repeated names in tool syntaxes

`JsonConstraint::ToolSyntax` gains:

- `call_separator`, and `block_close` without tagged arguments: calls share one
  block, as in `[f(k='v'), g()]`.
- `ValueSyntax::keyword_root`: root parameters as keyword arguments
  (`k='v', n=3`), with Python's `True`, `False` and `None` also admitted.
- `repeat_prefix`/`repeat_suffix`: the call writes its name a second time.

New fields default to empty or false; existing grammars are unchanged.
