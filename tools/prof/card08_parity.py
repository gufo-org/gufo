#!/usr/bin/env python3
"""Card 08 parity: sources that kept their path but changed CMake target.

Card 07 moved Qwen sources to new paths; Card 08 (RFC step 5) moved the MiniMax
H3 sources, the eval tier and the comparison helper out of `gufo_core`, and the
DeepSeek chat template and scalar sampler out of `gufo_deepseek_v4_flash` into
`gufo_deepseek_v4_flash_text`. Every source stayed where it is, so the question
is the mirror image: the same path must compile with the same effective flags
and emit the same code after the move.

The moved set is derived, not declared: `compile_commands.json` writes each
object below `CMakeFiles/<target>.dir/`, so a source whose object directory
changed owner is a moved source.

```sh
nix develop -c python3 tools/prof/card08_parity.py --mode flags \\
  --before build/card07-release/compile_commands.json \\
  --after build/card08-release/compile_commands.json
nix develop -c python3 tools/prof/card08_parity.py --mode object --before ... --after ...
nix develop -c python3 tools/prof/card08_parity.py --mode device --before ... --after ... \\
  --scratch /tmp/card08-parity
```

Any difference outside `ALLOWED` fails, and an `ALLOWED` entry that no longer
matches anything also fails, so a justification cannot outlive the difference
it explains.
"""

import argparse
import collections
import os
import shutil
import hashlib
import json
import re
import shlex
import subprocess
import struct
import tempfile
from pathlib import Path

DROP_VALUES = {"-o", "-c", "-MF", "-MT", "-MQ"}
# (sources) -> (removed flags, added flags, reason). Flags are compared with the
# repository root stripped, so entries stay portable across build machines.
ALLOWED = (
    (("src/eval/", "src/testing/compare/"),
     ('-DGUFO_FFMPEG_EXECUTABLE=', '-DGUFO_FFPROBE_EXECUTABLE='),
     (),
     "gufo_core carried the FFmpeg discovery macros for the H3 media sources only; "
     "eval and comparison sources never read them."),
    (("src/models/deepseek_v4_flash/chat_template.cpp",
      "src/models/deepseek_v4_flash/runtime/sampling.cpp"),
     ("src/models/deepseek_v4_flash/runtime",
      "src/models/deepseek_v4_flash/kernels/rocm",
      "src/models/deepseek_v4_flash/kernels/rocm/detail",
      "src/models/deepseek_v4_flash/kernels/rocm/mmq"),
     ("src/models/deepseek_v4_flash/text/../runtime", "-Wall", "-Wextra", "-Wpedantic"),
     "The text tier needs the runtime header directory and nothing else: the ROCm "
     "kernel, detail and mmq search paths belong to the HIP backend. The directory is "
     "written relative to the new target, and both files compiled with the project "
     "warnings as test-executable sources before."),
)


def load(path, root):
    """Read one compile database and reject a source compiled twice.

    CMake generates sources into the build directory itself (the per-executable
    extension registry), and that directory differs between the two databases,
    so anything below the entry's own compile directory is not a source under
    comparison. Every remaining source must appear exactly once: Card 08
    acceptance is one implementation per source, so a second target compiling
    the same file fails here instead of passing with a smaller comparison set.
    """
    entries = {}
    for entry in json.loads(Path(path).read_text()):
        source_path = Path(entry["file"]).resolve()
        # CMake compiles from its own build directory, so a source below that
        # directory is generated there (the per-executable extension registry),
        # whatever file the database was read from.
        if Path(entry["directory"]).resolve() in source_path.parents:
            continue
        source = str(source_path.relative_to(root))
        if source in entries:
            raise ValueError(f"{path}: duplicate compilation of {source}: "
                             f"{owner(entries[source])}, {owner(entry)}")
        entries[source] = entry
    return entries


def owner(entry):
    match = re.search(r"CMakeFiles/([^/]+)\.dir/", entry["output"])
    return match.group(1) if match else "unknown-target"


def flags(entry, root):
    result, skip = [], False
    for argument in shlex.split(entry["command"]):
        if skip:
            skip = False
            continue
        if argument in DROP_VALUES:
            skip = True
            continue
        if argument == entry["file"]:
            continue
        result.append(argument.replace(str(root) + "/", ""))
    return result


def classify_flags(source, removed, added):
    """Return the reason for an expected flag difference, or None if unexplained."""
    reasons = []
    for patterns, gone, extra, reason in ALLOWED:
        if not source.startswith(patterns):
            continue
        if any(not argument.startswith(tuple(gone)) for argument in removed):
            continue
        if any(not argument.startswith(tuple(extra)) for argument in added):
            continue
        reasons.append(reason)
    return reasons[0] if reasons else None


def unused_justifications(sources, differences):
    """Report ALLOWED entries that no current difference needs."""
    stale = []
    for index, (patterns, _, _, _) in enumerate(ALLOWED):
        covered = any(source.startswith(patterns) and source in differences
                      for source in sources)
        if not covered:
            stale.append(f"ALLOWED[{index}] for {', '.join(patterns)}")
    return stale


def section(path, name):
    data = Path(path).read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2:
        raise ValueError(f"{path}: expected little-endian ELF64")
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count, strings = struct.unpack_from("<HHH", data, 58)
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size)
               for i in range(count)]
    table = headers[strings]
    names = data[table[4]:table[4] + table[5]]
    for header in headers:
        start = header[0]
        if names[start:names.index(b"\0", start)].decode() == name:
            return data[header[4]:header[4] + header[5]]
    raise ValueError(f"{path}: missing {name}")


def find_bundler(explicit):
    """Return a bundler that can unpack the zstd ROCm bundle the HIP compiler emits.

    The `clang-offload-bundler` on PATH is the standalone clang-tools one; it needs
    `llvm-objcopy` and rejects the zstd bundles hipcc writes, so prefer the HIP
    toolchain copy the same way tools/ci/check-kernel-resources.py resolves it.
    """
    if explicit:
        return explicit
    hip_clang = os.environ.get("HIP_CLANG")
    candidates = []
    if hip_clang:
        candidates.append(Path(hip_clang).parent / "clang-offload-bundler")
    if shutil.which("hipconfig"):
        result = subprocess.run(["hipconfig", "-l"], capture_output=True, text=True,
                                timeout=30, check=False)
        if result.returncode == 0 and result.stdout.strip():
            candidates.append(Path(result.stdout.strip()) / "clang-offload-bundler")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return shutil.which("clang-offload-bundler") or "clang-offload-bundler"


def object_path(entry, base):
    """Locate an object recorded in compile_commands.json.

    CMake writes `output` relative to the top build directory for subdirectory
    targets and relative to `directory` for the top-level one, so both roots
    are tried before the path is reported as missing.
    """
    recorded = Path(entry["output"])
    candidates = ([recorded] if recorded.is_absolute()
                  else [Path(entry["directory"]) / recorded, base / recorded])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def device_images(entry, bundler, work, tag):
    obj = entry["object"]
    try:
        payload = section(obj, ".hip_fatbin")
    except ValueError:
        return None
    bundle, image = work / f"{tag}.bundle", work / f"{tag}.hsaco"
    bundle.write_bytes(payload)
    result = subprocess.run([bundler, "--unbundle", "--type=o",
                             f"--input={bundle}", f"--output={image}",
                             "--targets=hip-amdgcn-amd-amdhsa--gfx1151"],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{entry['file']}: {tag} unbundle failed with {bundler}: "
                         f"{result.stderr.strip()}\nnote: pass --bundler <hipClang "
                         f"clang-offload-bundler>; the PATH one cannot read zstd bundles.")
    return image


def symbols(obj):
    out = subprocess.run(["nm", "--defined-only", "--demangle", str(obj)],
                         capture_output=True, text=True, check=True).stdout
    return {line.split(" ", 2)[-1] for line in out.splitlines() if line.strip()}


def disasm(obj):
    """symbol -> list of (mnemonic, operand text); leading addresses dropped."""
    proc = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(obj)],
                          capture_output=True, text=True, check=True).stdout
    funcs, current, body = {}, None, []
    for line in proc.splitlines():
        if line and not line.startswith(" ") and line.endswith(">:") and "<" in line:
            if current:
                funcs[current] = body
            current = line[line.index("<") + 1:-2]
            body = []
        elif line.strip().endswith(":") and line.startswith(" "):
            continue
        elif current and line.strip():
            parts = line.strip().split(" ", 1)
            body.append((parts[0], parts[1].split("\t")[-1].strip() if len(parts) > 1 else ""))
    if current:
        funcs[current] = body
    return funcs


# Per-translation-unit identifiers the HIP compiler embeds: the CU id (also as a
# call target) and the device-binary handle. Both follow the object path.
CUID = re.compile(r"__hip_(?:cuid|gpubin_handle)_[0-9a-fA-F]+")


def classify_code(symbol, before, after):
    """None for equal code or a pure move artifact, otherwise the failure reason.

    Retargeting a translation unit changes only the per-TU `__hip_cuid_*`
    identifier and `__FILE__`-derived offsets in the local label pool.
    """
    if before == after:
        return None
    if len(before) != len(after):
        return "instruction count changed"
    if [mnemonic for mnemonic, _ in before] != [mnemonic for mnemonic, _ in after]:
        return "opcode or control flow changed"
    moved = [(one, two) for (_, one), (_, two) in zip(before, after) if one != two]
    if all(CUID.sub("__hip_cuid_*", one) == CUID.sub("__hip_cuid_*", two)
           for one, two in moved):
        return None
    if all(".L" in one or ".L" in two for one, two in moved):
        return None
    return f"operand changed outside the CU id/local label pool: {moved[0]}"[:120]


def compare_objects(label, before_object, after_object, counts, failures):
    """Compare one moved translation unit: exported symbols, then host code."""
    for obj in (before_object, after_object):
        if not obj.is_file():
            failures.append(f"{label}: object {obj} is missing")
            return
    one, two = symbols(before_object), symbols(after_object)
    masked_one = {CUID.sub("__hip_<tu-id>_", name) for name in one}
    masked_two = {CUID.sub("__hip_<tu-id>_", name) for name in two}
    if one == two:
        counts["symbols identical"] += 1
    elif masked_one == masked_two:
        counts["symbols identical apart from the TU id"] += 1
    else:
        failures.append(f"{label}: exported symbol set changed: "
                        f"-{sorted(one - two)[:3]} +{sorted(two - one)[:3]}")
        return
    changed = []
    before_code, after_code = disasm(before_object), disasm(after_object)
    for name in sorted(set(before_code) & set(after_code)):
        reason = classify_code(name, before_code[name], after_code[name])
        if reason:
            changed.append(f"{name}: {reason}")
    if changed:
        failures.append(f"{label}: {len(changed)} function(s) changed, first {changed[0]}")
    else:
        counts["code identical or move artifact"] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--before", type=Path, required=True,
                        help="parent revision compile_commands.json")
    parser.add_argument("--after", type=Path, required=True,
                        help="candidate compile_commands.json")
    parser.add_argument("--mode", choices=["flags", "object", "device"], required=True)
    parser.add_argument("--bundler", default=None)
    parser.add_argument("--scratch", type=Path,
                        help="device mode: where to unpack fatbins (removed after)")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        old, new = load(args.before, root), load(args.after, root)
        missing = sorted(old.keys() ^ new.keys())
        if missing:
            raise ValueError("sources present on one side only: " + ", ".join(missing))
    except ValueError as error:
        print(f"{args.mode}: FAIL {error}")
        return 1
    for entries, base in ((old, args.before.parent), (new, args.after.parent)):
        for entry in entries.values():
            entry["object"] = object_path(entry, base)
    moved = sorted(source for source, entry in old.items()
                   if source in new and owner(entry) != owner(new[source]))
    if args.mode == "device":
        moved = [source for source in moved if source.endswith((".hip", ".hip.cpp"))]
    if not moved:
        print(f"{args.mode}: no moved sources between {args.before} and {args.after}")
        return 1
    counts = collections.Counter()
    differences, failures = set(), []
    bundler = find_bundler(args.bundler)
    scratch = args.scratch or Path(__import__("tempfile").gettempdir())
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temporary:
        work = Path(temporary)
        for source in moved:
            before, after = old[source], new[source]
            label = f"{source} ({owner(before)} -> {owner(after)})"
            if args.mode == "flags":
                one, two = flags(before, root), flags(after, root)
                if one == two:
                    counts["flags identical"] += 1
                    continue
                removed = [flag for flag in one if flag not in two]
                added = [flag for flag in two if flag not in one]
                reason = classify_flags(source, removed, added)
                differences.add(source)
                if reason is None:
                    failures.append(f"{label}: unexplained flag difference "
                                    f"-{removed} +{added}")
                else:
                    counts["flags justified"] += 1
                    print(f"JUSTIFIED {label}")
                    for flag in removed:
                        print(f"    - {flag}")
                    for flag in added:
                        print(f"    + {flag}")
                continue
            before_object, after_object = before["object"], after["object"]
            if args.mode == "object":
                compare_objects(label, before_object, after_object, counts, failures)
                continue
            images = [device_images(item, bundler, work, tag)
                      for tag, item in (("before", before), ("after", after))]
            if bool(images[0]) != bool(images[1]):
                failures.append(f"{label}: device image exists on one side only")
                continue
            if not all(images):
                counts["no device image (host wrapper)"] += 1
                continue
            for name in (".text", ".rodata", ".note"):
                try:
                    values = [section(image, name) for image in images]
                except ValueError as error:
                    failures.append(f"{label}: {error}")
                    continue
                if values[0] != values[1] or not values[0]:
                    failures.append(f"{label}: device {name} differs")
                else:
                    counts[f"device {name} identical"] += 1
            print(f"DEVICE {label} " + " ".join(
                f"{name}={hashlib.sha256(section(image, name)).hexdigest()[:12]}"
                for name in (".text", ".rodata", ".note") for image in images[:1]))
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")
    stale = unused_justifications(moved, differences) if args.mode == "flags" else []
    for entry in stale:
        failures.append(f"{entry} matches no flag difference; delete it or its difference")
    print(f"{args.mode}: compared={len(moved)} failures={len(failures)}")
    for failure in failures:
        print(f"  FAIL {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
