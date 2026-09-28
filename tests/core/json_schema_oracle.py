#!/usr/bin/env python3
"""Compare the native byte grammar with the independent JSON Schema validator.

Run in nix develop after building json_constraint_test. Numeric probes use
canonical decimal notation and Decimal in the oracle, avoiding binary-float
multipleOf rounding and the grammar's intentional integer spelling policy.
"""

import argparse
from decimal import Decimal
import json
import itertools
import random
import subprocess

from jsonschema import Draft202012Validator, FormatChecker


def ecmascript_cases(node):
    """Use V8, not Python re, for ECMA-262 Unicode-mode pattern semantics."""
    words = ["".join(chars) for n in range(4)
             for chars in itertools.product("ab1_é\n", repeat=n)]
    words += ["\r", "\r\n", "\v", "\f", "\u0085", "\u2028", "\u2029",
              "\ufeff", "\u2003", "\u0301", "\u0661", "😀", "a😀", "a\n",
              "a\r", "a\r\n", "a\u2028", "a\u2029", "a\v", "a\u0085",
              "\0", "[", "&", "-", " cat!", "cats", "a cat b"]
    patterns = [r"^\w+$", r"^\W+$", r"^\d+$", r"^\D+$", r"^\s+$", r"^\S+$",
                "^.$", "^a$", "a$", "^a", "^a.*$", "^[^]$", "^[]*$",
                r"^[\w]+$", r"^[^\d]+$", r"^[\s]+$", r"^[\p{L}]+$",
                r"^[\p{Script=Greek}]+$", r"^[\P{L}]+$", r"^\uD83D\uDE00$",
                "^(?=a|b)(?:aa|b)$", "^(?!ab)[ab]+$", "^(?=.{2}$).*$",
                "(?:^a|b$)", "^(?:a?|b){1,3}$", "^a(?=b$)b$", r"^[a b]+$",
                r"\b", r"\B", r"\bcat\b", r"^\b[ab]+\b$", r"a\Bb",
                r"(?=\ba)a", r"^(?!\ba)[ab]+$", r"\Bé\B",
                r"^\u{1f600}$", r"^\cJ$", r"^\0$", r"^[\b\0]$",
                r"^[a&&b]$", r"^[[a]$"]
    requests = [{"pattern": pattern, "minimum": minimum, "maximum": maximum,
                 "words": words}
                for pattern in patterns for minimum, maximum in [(0, 4), (2, 3)]]
    reference = subprocess.run([node, "-e", r"""
const fs = require('fs');
const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(cases.map(c => {
  const regex = new RegExp(c.pattern, 'u');
  return c.words.map(s => [...s].length >= c.minimum &&
    [...s].length <= c.maximum && regex.test(s));
})));
"""], input=json.dumps(requests), text=True, capture_output=True,
        check=True, timeout=30)
    cases = []
    for request, expected in zip(requests, json.loads(reference.stdout)):
        schema = {"type": "object", "properties": {"x": {
            "type": "string", "pattern": request["pattern"],
            "minLength": request["minimum"], "maxLength": request["maximum"]}},
            "required": ["x"], "additionalProperties": False}
        # Both JSON spellings must implement the same decoded-string language.
        texts = ['{"x":' + json.dumps(word, ensure_ascii=escaped) + '}'
                 for word in words for escaped in (False, True)]
        cases.append((schema, texts, [valid for valid in expected for _ in range(2)]))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", help="Path to json_constraint_test")
    parser.add_argument("--node", help="Optional Node.js executable for the independent ECMA-262 oracle")
    args = parser.parse_args()
    rng = random.Random(283)
    cases = []

    def add(property_schema, values):
        schema = {"type": "object", "properties": {"x": property_schema},
                  "required": ["x"], "additionalProperties": False}
        # Preserve decimal JSON values in both schema and instance evaluation.
        teacher_schema = json.loads(json.dumps(schema), parse_float=Decimal)
        validator = Draft202012Validator(teacher_schema, format_checker=FormatChecker())
        texts = ['{"x":' + value + '}' for value in values]
        expected = [validator.is_valid(json.loads(text, parse_float=Decimal)) for text in texts]
        cases.append((schema, texts, expected))

    add({"type": "object", "properties": {"a": {"type": "integer"},
         "b": {"type": "boolean"}}, "required": ["a", "b"],
         "additionalProperties": False,
         "enum": [{"b": True, "a": 1}, {"a": 2, "b": False}]},
        [json.dumps({"a": a, "b": b}) for a in (0, 1, 2) for b in (False, True)])
    add({"type": "array", "items": {"type": "integer", "minimum": 0},
         "minItems": 2, "enum": [[1, 2], [-1], [2], [0, 3]]},
        ["[]", "[1,2]", "[-1]", "[2]", "[0,3]", "[1,3]"])
    add({"type": "string", "enum": ["a", "b"], "const": "b"},
        ['"a"', '"b"', '"c"'])
    add({"type": "object", "additionalProperties": False,
         "enum": [{"unexpected": 1}, {}]}, ['{}', '{"unexpected":1}'])
    add({"type": "array", "items": {"type": "integer", "minimum": 2, "maximum": 1},
         "enum": [[1], []]}, ['[]', '[1]', '[2]'])
    add({"anyOf": [{"type": "string", "pattern": "^a$", "minLength": 2},
                  {"type": "string", "const": "b"}]},
        ['""', '"a"', '"aa"', '"b"', '"c"'])
    for _ in range(40):
        low, high = sorted(rng.sample(range(-10, 11), 2))
        property_schema = {"$ref": "#/$defs/N", "minimum": low, "maximum": high}
        schema = {"type": "object", "properties": {"x": property_schema},
                  "required": ["x"], "additionalProperties": False,
                  "$defs": {"N": {"type": "integer", "minimum": -5, "maximum": 5}}}
        values = [json.dumps({"x": i}) for i in range(-12, 13)]
        validator = Draft202012Validator(schema)
        expected = [validator.is_valid(json.loads(text)) for text in values]
        if any(expected):
            cases.append((schema, values, expected))
    add({"anyOf": [{"type": "integer", "minimum": 1},
                  {"type": "integer", "minimum": 10}], "maximum": 5},
        [str(i) for i in range(12)])
    for a, b in ((.2, .3), (.25, .4), (1.2, .18), (2, 3), (10, 25), (1e-4, 2e-4)):
        add({"$defs": {"N": {"type": "number", "multipleOf": a}},
             "$ref": "#/properties/x/$defs/N", "multipleOf": b,
             "minimum": -2, "maximum": 2},
            [str(Decimal(i) / 100) for i in range(-210, 211)])
    for a, b in (("hostname", "ipv4"), ("ipv4", "hostname"),
                 ("hostname", "uuid"), ("hostname", "date")):
        add({"$defs": {"S": {"type": "string", "format": a}},
             "$ref": "#/properties/x/$defs/S", "format": b,
             "pattern": "^[0-9]", "maxLength": 40},
            [json.dumps(s) for s in ("127.0.0.1", "example.com", "-bad",
             "256.0.0.1", "2024-02-29", "2023-02-29", "1:::2",
             "12345678-1234-1234-1234-123456789abc")])
    add({"$defs": {"S": {"type": "string", "format": "date"}},
         "anyOf": [{"$ref": "#/properties/x/$defs/S", "format": "ipv4"},
                   {"type": "string", "const": "fallback"}]},
        ['"2024-02-29"', '"127.0.0.1"', '"fallback"'])

    for integer in (False, True):
        for _ in range(100):
            low, high = sorted(rng.sample(range(-40, 41), 2))
            scale = 1 if integer else 10
            schema = {"type": "integer" if integer else "number",
                      rng.choice(["minimum", "exclusiveMinimum"]): low / scale,
                      rng.choice(["maximum", "exclusiveMaximum"]): high / scale}
            if rng.randrange(2):
                schema["multipleOf"] = rng.choice([.1, .25, .5, 1, 1.5, 3])
            values = [str(i) if integer else str(Decimal(i) / 10) for i in range(-50, 51)]
            add(schema, values)
    for limit in (1, 2, 3, 10, 257, 1_000_000_000):
        for minimum in (0, 1, limit):
            add({"type": "array", "items": {"type": "boolean"},
                 "minItems": minimum, "maxItems": limit},
                [json.dumps([True] * count) for count in (0, 1, 2, 3, 10, 256, 257, 258)])
    words = ["", "a", "ab", "abc", "c0", "abc1", "c5abc3", "ABC", "aABC0",
             "é", "é😀", "e\u0301😀", "@name_12", "x@name", "\n", "a b", "a&b"]
    for pattern in ("^@[a-zA-Z0-9_]+$", "^(?:ab|c[0-9])+$", "[A-Z]",
                    "^(?=.*[A-Z])(?=.*[0-9]).+$", "^é😀$", "^OK$",
                    r"^[a-zA-Z0-9_]+$", r"^(?:(?:ab){2}|c)$", r"^[a b]+$", r"^[a&b]+$"):
        for minimum, maximum in ((0, 100), (2, 4)):
            add({"type": "string", "pattern": pattern, "minLength": minimum,
                 "maxLength": maximum},
                [json.dumps(word, ensure_ascii=ascii_only)
                 for word in words for ascii_only in (True, False)])
    short_words = ["".join(chars) for n in range(5)
                   for chars in itertools.product("abcé", repeat=n)]
    for pattern in ("^(?:(?:ab){2}|c)$", "^(?:ab)+$", "^(?:[^a-c]{4}|c)$",
                    "^a{1,3}b?$", "^a*?b+$", "a|b$", r"^[a-zA-Z0-9_]{1,3}$",
                    "^.{2,4}$", r"^[^ab]{1,2}$", "^(a|bc){1,2}$",
                    "^(?=a|c)(?:(?:ab){2}|c)$", "^(?!ab)(?:ab|ac|b)$",
                    "^(?=.{2}$)(?!aa)[ab]+$", "^(?:a(?=b)b|c)$",
                    "^a(?=b$)b$", "^(?!.*bb)[ab]{1,4}$",
                    "^(?:(?=a)a){2}$", "(?:^ab|c$)", "^(?:a?|b){2,4}$",
                    "^(?!.*bb)(?:.{1,3}|(?:.{0,2}[ab]{0,2}|(?:ab|b)?[^a]{0,2}){0,2}[^a]{0,2}){1,3}(?:a?|b){0,2}$"):
        for minimum, maximum in ((0, 3), (2, 4), (3, 3)):
            add({"type": "string", "pattern": pattern, "minLength": minimum,
                 "maxLength": maximum},
                [json.dumps(word, ensure_ascii=False) for word in short_words])
    for format_name, examples in (
        ("date", ["2024-02-29", "2023-02-29", "2000-02-29", "1900-02-29", "2026-04-31"]),
        ("ipv4", ["127.0.0.1", "192.168.1.89", "256.1.2.3", "01.2.3.4"]),
        ("ipv6", ["::", "::1", "2001:db8::1", "::ffff:192.0.2.1", "1:::2", "1:2:3:4:5:6:7:8:9"]),
        ("uuid", ["12345678-1234-1234-1234-123456789abc", "12345678-1234-1234-1234-123456789abz"]),
    ):
        add({"type": "string", "format": format_name}, [json.dumps(value) for value in examples])
    # Exhaustive finite-language prefix oracle: checking only complete values
    # misses prefixes which are admitted but cannot finish within maxLength.
    prefix_cases = []
    alphabet_words = ["".join(chars) for n in range(5)
                      for chars in itertools.product("abc", repeat=n)]
    for pattern in ("^(?:(?:ab){2}|c)$", "^(?:ab)+$", "^(?:a{2,3}|b[ac])$",
                    "^(?:a{500}|b{1,500})$", "^(?=a|c)(?:(?:ab){2}|c)$",
                    "^(?!ab)(?:ab|ac|b)$", "^(?=.{2}$)(?!aa)[ab]+$",
                    "^(?:a(?=b)b|c)$", "^a(?=b$)b$", "^(?!.*bb)[ab]{1,4}$",
                    "^(?:(?=a)a){2}$", "^(?:a?|b){2,4}$"):
        # Letter-only finite languages also agree with Python's regex dialect.
        for minimum, maximum in ((0, 1), (0, 2), (0, 3), (0, 4)):
            schema = {"type": "object", "properties": {
                "x": {"type": "string", "pattern": pattern, "minLength": minimum,
                      "maxLength": maximum}}, "required": ["x"], "additionalProperties": False}
            validator = Draft202012Validator(schema)
            language = [word for word in alphabet_words if validator.is_valid({"x": word})]
            prefixes = ['{"x":"' + word for word in alphabet_words]
            expected = [any(value.startswith(word) for value in language)
                        for word in alphabet_words]
            prefix_cases.append((schema, prefixes, expected))
    if args.node:
        cases.extend(ecmascript_cases(args.node))
    records = "\n".join(json.dumps({"schema": schema, "texts": texts}) for schema, texts, _ in cases)
    records += "\n" + "\n".join(json.dumps({"schema": schema, "texts": [], "prefixes": prefixes})
                              for schema, prefixes, _ in prefix_cases)
    result = subprocess.run([args.probe, "--probe"], input=records + "\n", text=True,
                            capture_output=True, check=True, timeout=120)
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(outputs) == len(cases) + len(prefix_cases), (len(outputs), len(cases), result.stderr)
    decisions = 0
    for (schema, texts, expected), output in zip(cases, outputs):
        if "error" in output:
            assert not any(expected), (schema, output)
            continue  # Rejecting an empty numeric interval is expected.
        assert len(output["accepted"]) == len(expected), (schema, output)
        for text, actual, wanted in zip(texts, output["accepted"], expected):
            assert actual == wanted, (schema, text, actual, wanted)
            decisions += 1
    prefix_decisions = 0
    for (schema, prefixes, expected), output in zip(prefix_cases, outputs[len(cases):]):
        if "error" in output:
            assert not any(expected), (schema, output)
            continue
        assert output["viable"] == expected, (schema, [
            prefix for prefix, actual, wanted in zip(prefixes, output["viable"], expected)
            if actual != wanted])
        prefix_decisions += len(prefixes)
    print(f"JSON Schema oracle: {decisions} acceptance decisions across {len(cases)} schemas passed")
    print(f"JSON Schema prefix oracle: {prefix_decisions} completion-feasibility decisions passed")


if __name__ == "__main__":
    main()
