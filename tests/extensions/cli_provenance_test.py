#!/usr/bin/env python3
"""Check the actual CLI provenance contract without model loads or GPU calls."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

binary = sys.argv[1]
version = subprocess.check_output([binary, "--version"], text=True)
assert version.splitlines()[0].startswith("gufo version "), version
report = json.loads(subprocess.check_output([binary, "diagnose", "--section", "extensions", "--json"], text=True))
assert f'Extension API: {report["extensionApiVersion"]}' in version
assert f'Extension selection: {report["selectionMode"]}' in version
for extension in report["builtExtensions"]:
    assert f'{extension["name"]} {extension["version"]} ({extension["source"]})' in version
for extension in report["excludedExtensions"]:
    assert f'excluded {extension["name"]} {extension["version"]} ({extension["source"]}): {extension["reason"]}' in version
assert subprocess.check_output([binary, "--version"], text=True) == version
assert subprocess.check_output([binary, "diagnose", "--section", "extensions"], text=True) == "\n".join(version.splitlines()[1:]) + "\n"
with tempfile.TemporaryDirectory() as temp:
    output = Path(temp) / "extensions.json"
    printed = subprocess.check_output([binary, "diagnose", "--section", "extensions", "--json", "--output", str(output)], text=True)
    assert json.loads(output.read_text()) == json.loads(printed) == report
# Preserve existing hardware schema and section behavior.
platform = subprocess.run([binary, "diagnose", "--section", "platform", "--json"], text=True, capture_output=True)
assert platform.returncode in (0, 1), platform.stderr
assert "builtExtensions" not in json.loads(platform.stdout)
print(f'PASS: actual gufo CLI provenance ({report["selectionMode"]})')
