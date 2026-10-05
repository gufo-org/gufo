#!/usr/bin/env python3
"""Write the schema/output cases of the llama.cpp tool-grammar goldens.

Usage: cases.py OUTPUT_DIR  ->  cases_qwen.tsv and cases_deepseek.tsv, one row
per tool schema: label, parameters JSON, then four candidate outputs (a valid
value, an invalid value, no arguments, an undeclared argument).
"""

import json
from pathlib import Path
import sys


def qwen(args):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args)
    return f"<tool_call>\n<function=record>\n{body}</function>\n</tool_call>"


def deepseek(args):
    body = ""
    for k, v in args:
        try:
            string = isinstance(json.loads(v), str)
        except ValueError:
            string = True
        flag = "true" if string else "false"
        body += (f'<\uff5cDSML\uff5cparameter name="{k}" string="{flag}">{v}'
                 f'</\uff5cDSML\uff5cparameter>\n')
    return (f'<\uff5cDSML\uff5ctool_calls>\n<\uff5cDSML\uff5cinvoke name="record">\n{body}'
            f'</\uff5cDSML\uff5cinvoke>\n</\uff5cDSML\uff5ctool_calls>')


props = [
 ("string", {"type":"string"}, "plain text", None),
 ("pattern", {"type":"string","pattern":"^[0-9]{4}$"}, "2026", "not a year"),
 ("pattern_tag", {"type":"string","pattern":"^\\n</parameter>$"}, "anything", None),
 ("format", {"type":"string","format":"uri"}, "https://a.b", "not a uri"),
 ("enum", {"type":"string","enum":["red","blue"]}, "red", "green"),
 ("minlen", {"type":"string","minLength":3}, "abcd", "a"),
 ("const_tag", {"type":"string","const":"\n</parameter>\n"}, "x", None),
 ("int_bounds", {"type":"integer","exclusiveMinimum":0,"maximum":9007199254740991}, "240000", "x"),
 ("int_empty", {"type":"integer","minimum":5,"maximum":2}, "5", "x"),
 ("one_of", {"oneOf":[{"type":"string"},{"type":"integer"}]}, "42", None),
 ("any_int_null", {"anyOf":[{"type":"integer"},{"type":"null"}]}, "null", "x"),
 ("any_enum_null", {"anyOf":[{"type":"string","enum":["a","b"]},{"type":"null"}]}, "a", "c"),
 ("all_of_str", {"allOf":[{"type":"string"},{"minLength":1}]}, "42", None),
 ("all_of_int", {"allOf":[{"type":"integer"},{"minimum":1}]}, "2", "x"),
 ("not_null", {"not":{"type":"null"}}, "anything", None),
 ("str_null", {"type":["string","null"]}, "null", None),
 ("object", {"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}, '{"x": 1}', '{"x": "a"}'),
 ("array", {"type":"array","items":{"type":"object","properties":{"k":{"type":"string"}},"required":["k"]}}, '[{"k": "v"}]', '[{}]'),
 ("untyped", {}, "free", None),
]
roots = {
 "plain": {"type":"object"},
 "dollar_schema": {"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object"},
 "addl_true": {"type":"object","additionalProperties":True},
 "pattern_props": {"type":"object","patternProperties":{"^x_":{"type":"integer"}}},
}
inline = {"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}
root_shapes = {
 "root_oneof": {"type":"object","oneOf":[inline]},
 "root_allof": {"type":"object","allOf":[inline]},
 "root_anyof": {"anyOf":[inline]},
 "root_ref": {"$ref":"#/$defs/R","$defs":{"R":inline}},
 "root_typed_ref": {"type":"object","$ref":"#/$defs/R","$defs":{"R":inline}},
 "root_const": {"type":"object","const":{"value":"alpha"}},
 "root_props_allof": {"type":"object","properties":{"value":{"type":"string"}},"allOf":[{"required":["value"]}]},
 "root_if_then": {"type":"object","properties":{"kind":{"type":"string"}},"required":["kind"],"if":{"properties":{"kind":{"const":"x"}}},"then":{"properties":{"value":{"type":"integer"}}}},
 "root_map": {"type":"object","additionalProperties":{"type":"string"}},
}


def rows(call):
    result = []
    for rname, root in roots.items():
        for pname, p, good, bad in props:
            schema = dict(root)
            schema["properties"] = {"v": p}
            schema["required"] = ["v"]
            cands = [call([("v", good)]), call([("v", bad if bad else good)]),
                     call([]), call([("v", good), ("zz", "1")])]
            result.append([f"{rname}/{pname}", json.dumps(schema), *map(json.dumps, cands)])
    for name, schema in root_shapes.items():
        cands = [call([("value", "alpha")]), call([("kind", "x"), ("value", "1")]),
                 call([]), call([("value", "alpha"), ("zz", "1")])]
        result.append([name, json.dumps(schema), *map(json.dumps, cands)])
    # The delimiter must start after the raw value. A Qwen value ending in
    # this prefix used to complete a delimiter across that boundary and admit
    # duplicated closers, even though llama.cpp's parser rejects the call.
    schema = {"type": "object", "properties": {"v": {"type": "string"}},
              "required": ["v"], "additionalProperties": False}
    overlap = "\n</parameter>" if call is qwen else "</｜DSML｜parameter>"
    cands = [call([("v", "archive.txt\n")]), call([("v", "archive.txt" + overlap)]),
             call([]), call([("v", "archive.txt"), ("zz", "1")])]
    result.append(["delimiter_overlap", json.dumps(schema), *map(json.dumps, cands)])
    return result


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    for name, call in (("qwen", qwen), ("deepseek", deepseek)):
        (directory / f"cases_{name}.tsv").write_text(
            "\n".join("\t".join(row) for row in rows(call)) + "\n")
