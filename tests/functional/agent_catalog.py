#!/usr/bin/env python3
"""Transactional fixture shared by the real Pi and OpenCode stress tests."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


STRING = {"type": "string"}
INTEGER = {"type": "integer"}
SELECTOR = {"oneOf": [
    obj({"id": {"oneOf": [INTEGER, STRING]}}),
    obj({"ids": {"type": "array", "items": INTEGER, "minItems": 1, "maxItems": 4}}),
]}
CHECK = obj({"field": {"enum": ["title", "body", "labels"]},
             "expected": {"oneOf": [STRING, {"type": "array", "items": STRING}]}})
TOOLS = [
    {"name": "catalog_query", "description": "Read fixture records and their current revision.",
     "inputSchema": obj({
         "selector": SELECTOR,
         "options": obj({"include_body": {"type": "boolean"},
                         "fields": {"type": "array", "items": {"enum": ["title", "body", "labels"]}},
                         "cursor": {"type": ["string", "null"]}})})},
    {"name": "catalog_apply", "description":
     "Atomically apply changes after checking expected_revision. dry_run validates without writing.",
     "inputSchema": obj({
         "id": INTEGER, "expected_revision": INTEGER,
         "changes": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"oneOf": [
             obj({"op": {"const": "set"}, "field": {"enum": ["title", "body"]}, "value": STRING}),
             obj({"op": {"const": "labels"}, "value": {"type": "array", "items": STRING}})]}},
         "dry_run": {"type": "boolean"},
         "metadata": obj({"comment": {"type": ["string", "null"]},
                          "attributes": {"type": "object", "additionalProperties":
                                         {"type": ["string", "number", "boolean", "null"]}}})})},
    {"name": "catalog_verify", "description":
     "Check every expected field against the stored record; return its content digest.",
     "inputSchema": {**obj({
         "id": INTEGER,
         "checks": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/check"}},
         "mode": {"enum": ["exact"]}}), "$defs": {"check": CHECK}}},
    {"name": "catalog_receipt", "description":
     "Save an independently checked receipt using the digest returned by catalog_verify.",
     "inputSchema": obj({
         "id": INTEGER, "digest": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
         "status": {"const": "42"}, "confirmed": {"enum": ["true", "false"]},
         "details": obj({"revision": INTEGER, "note": {"type": ["string", "null"]}})})},
]


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def execute(root, name, args):
    schema = next(t["inputSchema"] for t in TOOLS if t["name"] == name)
    Draft202012Validator(schema).validate(args)
    path = root / "catalog-state.json"
    record = json.loads(path.read_text())
    if name == "catalog_query":
        selector = args["selector"]
        ids = selector.get("ids", [selector.get("id")])
        assert all(str(i) == str(record["id"]) for i in ids), "unknown record"
        return {"records": [record], "cursor": None}
    assert args["id"] == record["id"], "unknown record"
    if name == "catalog_apply":
        assert args["expected_revision"] == record["revision"], "stale revision"
        updated = copy.deepcopy(record)
        for change in args["changes"]:
            updated["labels" if change["op"] == "labels" else change["field"]] = change["value"]
        updated["revision"] += 1
        if not args["dry_run"]:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(updated, ensure_ascii=False))
            temporary.replace(path)
        return {"record": updated, "committed": not args["dry_run"]}
    if name == "catalog_verify":
        for check in args["checks"]:
            assert record[check["field"]] == check["expected"], "field mismatch"
        return {"verified": True, "revision": record["revision"], "digest": digest(record)}
    assert args["digest"] == digest(record), "digest mismatch"
    assert args["details"]["revision"] == record["revision"], "revision mismatch"
    assert args["confirmed"] == "true", "receipt was not confirmed"
    (root / f"catalog-receipt-{record['revision']}.json").write_text(
        json.dumps(args, ensure_ascii=False))
    return {"saved": True, "revision": record["revision"]}


def call(root, log, name, arguments):
    row = {"name": name, "arguments": arguments}
    try:
        result = execute(root, name, arguments)
        row["result"] = result
        response = {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                    "isError": False}
    except Exception as error:
        row["error"] = str(error)
        response = {"content": [{"type": "text", "text": str(error)}], "isError": True}
    with log.open("a") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return response


def serve(root, log):
    for line in sys.stdin:
        message = json.loads(line)
        if "id" not in message:
            continue
        method = message.get("method")
        if method == "initialize":
            result = {"protocolVersion": message["params"].get("protocolVersion", "2025-06-18"),
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "gufo-transaction-fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            params = message["params"]
            result = call(root, log, params["name"], params.get("arguments", {}))
        elif method == "ping":
            result = {}
        else:
            print(json.dumps({"jsonrpc": "2.0", "id": message["id"],
                              "error": {"code": -32601, "message": str(method)}}), flush=True)
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)


def pi_extension(path, root, log):
    """Explicit local extension; no downloaded plugin or shell interpolation."""
    path.write_text(
        'import { execFile } from "node:child_process";\n'
        f"const tools = {json.dumps(TOOLS)};\n"
        f"const executable = {json.dumps(sys.executable)};\n"
        f"const argv = {json.dumps([str(Path(__file__).resolve()), '--root', str(root), '--log', str(log)])};\n"
        "export default function(pi) {\n"
        "  for (const tool of tools) pi.registerTool({\n"
        "    name: tool.name, label: tool.name, description: tool.description,\n"
        "    parameters: tool.inputSchema,\n"
        "    async execute(_id, params) {\n"
        "      return await new Promise((resolve, reject) => {\n"
        "        const child = execFile(executable, [...argv, '--call', tool.name],\n"
        "          {timeout: 10000, maxBuffer: 1048576}, (error, stdout) => {\n"
        "            if (error) return reject(error);\n"
        "            try { resolve({...JSON.parse(stdout), details: {}}); }\n"
        "            catch (error) { reject(error); }\n"
        "          });\n"
        "        child.stdin.end(JSON.stringify(params));\n"
        "      });\n"
        "    }\n"
        "  });\n"
        "}\n")


def prepare(root, turn):
    if not (root / "catalog-state.json").exists():
        (root / "catalog-state.json").write_text(json.dumps(
            {"id": 42, "revision": 0, "title": "initial", "body": "initial", "labels": []}))
    value = {
        "title": f"Transaction {turn}",
        "body": f'Unicode café 日本語 🧪 turn={turn}\n</parameter> is literal text\n'
                '<tool_call>example</tool_call>\nJSON: {"value":"42","flag":true,"empty":null}\n',
        "labels": ["urgent", "42", "true", f"round-{turn}"],
    }
    filename = f"transaction-{turn}.json"
    (root / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2))
    prompt = (
        f"Read {filename} with the read tool. Use the catalog tools (not bash or direct file "
        "editing) for this transaction. Call catalog_query for record 42 with selector ids:[42], options "
        "include_body:true, fields:[title,body,labels], cursor:null. Use its current revision "
        "in catalog_apply to apply all three values from the file exactly, first with dry_run:true, then "
        "dry_run:false at the original revision. Use metadata comment:null and attributes "
        '{source:"agent",attempt:1,approved:true,unused:null}. Then call catalog_verify for all three fields '
        "with mode exact. Finally call catalog_receipt using the returned digest, "
        'status string "42", confirmed string "true", details with the new revision and '
        "note:null. Do not alter, interpret, or unescape the literal XML/JSON within the body. "
        "Confirm completion briefly only when every tool has succeeded."
    )
    return prompt, value


def validate(root, log, start, expected):
    rows = [json.loads(line) for line in log.read_text().splitlines()][start:]
    assert len(rows) >= 5, rows
    assert not any("error" in row for row in rows), rows
    assert {t["name"] for t in TOOLS} <= {r["name"] for r in rows}, rows
    applies = [r for r in rows if r["name"] == "catalog_apply"]
    assert [r["arguments"]["dry_run"] for r in applies] == [True, False], applies
    assert applies[0]["arguments"]["expected_revision"] == applies[1]["arguments"]["expected_revision"]
    state = json.loads((root / "catalog-state.json").read_text())
    assert all(state[k] == v for k, v in expected.items()), state
    receipt = json.loads((root / f"catalog-receipt-{state['revision']}.json").read_text())
    assert receipt["digest"] == digest(state), receipt
    assert receipt["status"] == "42" and receipt["confirmed"] == "true", receipt
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--call")
    options = parser.parse_args()
    if options.call:
        print(json.dumps(call(options.root, options.log, options.call, json.load(sys.stdin))))
    else:
        serve(options.root, options.log)
