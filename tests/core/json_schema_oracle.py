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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", help="Path to json_constraint_test")
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
             "é", "é😀", "e\u0301😀", "@name_12", "x@name", "\n", "OK\n"]
    for pattern in ("^@[a-zA-Z0-9_]+$", "^(?:ab|c[0-9])+$", "[A-Z]",
                    "^(?=.*[A-Z])(?=.*[0-9]).+$", "^é😀$", "^OK$",
                    r"^[\w]+$", r"^[\s]+$", r"^(?:(?:ab){2}|c)$"):
        for minimum, maximum in ((0, 100), (2, 4)):
            add({"type": "string", "pattern": pattern, "minLength": minimum,
                 "maxLength": maximum},
                [json.dumps(word, ensure_ascii=ascii_only)
                 for word in words for ascii_only in (True, False)])
    short_words = ["".join(chars) for n in range(5)
                   for chars in itertools.product("abcé", repeat=n)]
    for pattern in ("^(?:(?:ab){2}|c)$", "^(?:ab)+$", "^(?:[^a-c]{4}|c)$",
                    "^a{1,3}b?$", "^a*?b+$", "a|b$", r"^[\w]{1,3}$",
                    "^.{2,4}$", r"^[^ab]{1,2}$", "^(a|bc){1,2}$"):
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
                    "^(?:a{500}|b{1,500})$"):
        # minLength=0 makes optional terminal line breaks irrelevant to these
        # letter-only prefixes (ICU and Python differ in '$' line-break rules).
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
