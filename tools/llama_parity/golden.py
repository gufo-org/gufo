#!/usr/bin/env python3
"""Join llama.cpp and gufo verdicts into tests/fixtures/llama_cpp_tool_grammar.json.

Usage: golden.py DIR LLAMA_REVISION OUTPUT. DIR holds cases_{qwen,deepseek}.tsv,
llama_{qwen,deepseek}.tsv (llama_probe) and gufo_{qwen,deepseek}.tsv
(gufo_probe ... required). Every disagreement must match a reviewed reason.
"""

from collections import Counter
import json
import sys

S, revision, output = sys.argv[1:4]
classes = {"cand1": "valid", "cand2": "invalid_value", "cand3": "missing_required", "cand4": "undeclared"}
def cases(path):
    d = {}
    for line in open(path):
        cols = line.rstrip("\n").split("\t")
        d[cols[0]] = (cols[1], [json.loads(c) for c in cols[2:]])
    return d
def verdicts(path, llama):
    d = {}
    for line in open(path):
        x = line.rstrip("\n").split("\t")
        if len(x) >= 3 and x[1].startswith("cand"):
            d[(x[0], x[1])] = (x[-1].split("=")[1] if llama else x[2]) == "yes"
    return d
rows, unexplained = [], []
for fmt in ("qwen", "deepseek"):
    cfile, lfile, gfile = f"cases_{fmt}.tsv", f"llama_{fmt}.tsv", f"gufo_{fmt}.tsv"
    cs, ll, gg = cases(f"{S}/{cfile}"), verdicts(f"{S}/{lfile}", True), verdicts(f"{S}/{gfile}", False)
    for label, (schema, cands) in cs.items():
        for i, text in enumerate(cands, 1):
            key = (label, f"cand{i}"); cls = classes[key[1]]
            if label.split("/")[-1] in ("one_of", "pattern_tag", "untyped", "not_null", "str_null", "all_of_str", "string", "format", "pattern") and cls == "invalid_value":
                cls = "valid"  # these candidates repeat a schema-valid value
            lv, gv = ll[key], gg[key]
            prop = label.split("/")[-1]
            note = ""
            if lv != gv:
                if gv and not lv and prop in ("all_of_int", "int_empty") and cls == "valid":
                    note = "llama.cpp's grammar rejects this schema-valid value"
                elif label in ("root_ref", "root_typed_ref"):
                    note = "gufo follows a root $ref, as its native path did before"
                elif not gv and lv and cls == "invalid_value":
                    note = "gufo enforces string const/enum/length, as its native path did before"
                elif not gv and lv and prop == "const_tag":
                    note = "gufo enforces the string const, as its native path did before"
                elif fmt == "deepseek" and gv and not lv and cls == "undeclared":
                    note = "DeepSeek string flag carries wildcard parameters on open roots, as before"
                elif fmt == "deepseek" and gv and not lv and label in ("root_if_then", "root_oneof", "root_allof"):
                    note = "DeepSeek open root admits wildcard parameters, as before"
                elif fmt == "deepseek" and gv and not lv and prop in ("one_of", "str_null", "untyped", "not_null"):
                    note = "DeepSeek also admits the typed string=\"false\" form, as before"
                elif fmt == "deepseek" and gv and not lv and prop in ("all_of_int", "int_empty"):
                    note = "llama.cpp's grammar rejects this schema-valid value"
                else:
                    unexplained.append((fmt, label, cls, lv, gv))
            rows.append({"format": fmt, "case": label, "parameters": json.loads(schema), "output": text,
                         "class": cls, "llama_cpp": lv, "gufo": gv, **({"difference": note} if note else {})})
assert not unexplained, unexplained
fixture = {"source": f"llama.cpp {revision} common/parsers/qwen3-coder.cpp and deepseek.cpp, tool_choice required; "
           "verdicts from its GBNF grammar engine (Qwen3.8 Flash-Next and DeepSeek V4 Flash templates)",
           "rows": rows}
with open(output, "w") as out:
    # One row per line keeps the fixture reviewable.
    out.write('{"source": ' + json.dumps(fixture["source"], ensure_ascii=False) + ',\n "rows": [\n')
    out.write(",\n".join("  " + json.dumps(row, ensure_ascii=False) for row in rows))
    out.write("\n ]\n}\n")
print(len(rows), "rows;", Counter(r.get("difference", "same") for r in rows))
