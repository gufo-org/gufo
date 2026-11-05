#!/usr/bin/env python3
"""Reproduce Card 07 compile flags and gfx1151 device section parity.

Device comparison includes all code/constants/resource notes per translation unit.
Any changed section fails; empty or missing device images cannot pass.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import struct
import shlex
import tempfile
from pathlib import Path


def find_bundler(explicit):
    """Return a bundler that can decompress ROCm offload bundles.

    The plain dev-shell `clang-offload-bundler` rejects zstd bundles, so prefer
    the HIP toolchain bundler the same way tools/ci/check-kernel-resources.py
    does, and keep --bundler as the explicit override.
    """
    if explicit:
        return explicit
    if os.environ.get('HIP_CLANG'):
        candidate = os.path.join(os.path.dirname(os.environ['HIP_CLANG']),
                                 'clang-offload-bundler')
        if os.path.isfile(candidate):
            return candidate
    hipconfig = shutil.which('hipconfig')
    if hipconfig:
        result = subprocess.run([hipconfig, '-l'], capture_output=True, text=True,
                                timeout=30, check=False)
        if result.returncode == 0 and result.stdout.strip():
            candidate = os.path.join(result.stdout.strip(), 'clang-offload-bundler')
            if os.path.isfile(candidate):
                return candidate
    return shutil.which('clang-offload-bundler') or 'clang-offload-bundler'


def section(path, name):
    data = Path(path).read_bytes()
    if data[:5] != b'\x7fELF\x02' or data[5] != 1:
        raise ValueError(f'{path}: expected little-endian ELF64')
    offset = struct.unpack_from('<Q', data, 40)[0]
    size, count, strings = struct.unpack_from('<HHH', data, 58)
    headers = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * size)
               for i in range(count)]
    table = headers[strings]
    names = data[table[4]:table[4] + table[5]]
    for header in headers:
        start = header[0]
        if names[start:names.index(b'\0', start)].decode() == name:
            return data[header[4]:header[4] + header[5]]
    raise ValueError(f'{path}: missing {name}')


def mapped(source):
    prefix = 'src/models/qwen/'
    relative = source.removeprefix(prefix)
    for old, new in [('hip/kernels/', 'kernels/'), ('vision/', 'vision/')]:
        if relative.startswith(old):
            return 'src/families/qwen/' + new + relative[len(old):]
    if relative in ('tokenizer.cpp', 'chat_template.cpp'):
        return 'src/families/qwen/text/' + relative
    return source


def flags(entry, root):
    args = entry.get('arguments') or shlex.split(entry['command'])
    result, skip = [], False
    for arg in args[1:]:
        if skip:
            skip = False
            continue
        if arg in ('-o', '-c', '-MF', '-MT', '-MQ'):
            skip = True
            continue
        # Keep include/define/options intact; only compilation paths vary.
        # Executable discovery macros belonged to core, never Qwen arithmetic.
        if arg.startswith(('-DGUFO_FFMPEG_EXECUTABLE=', '-DGUFO_FFPROBE_EXECUTABLE=')):
            continue
        if arg == entry['file']:
            continue
        result.append(arg)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--mode', choices=['flags', 'device'], required=True)
    parser.add_argument('--bundler', default=None,
                        help='offload bundler (default: the HIP toolchain bundle)')
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    args.bundler = find_bundler(args.bundler)
    root = args.root.resolve()
    def load(path):
        return {str(Path(e['file']).relative_to(root)): e for e in json.loads(path.read_text())}
    old, new = load(args.before), load(args.after)
    rows, failed = 0, 0
    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.scratch) as temporary:
        work = Path(temporary)
        for source, entry in sorted(old.items()):
            if not source.startswith('src/models/qwen/'):
                continue
            if args.mode == 'device' and not source.endswith('.hip'):
                continue
            other = new[mapped(source)]
            rows += 1
            if args.mode == 'flags':
                equal = flags(entry, root) == flags(other, root)
                print(source, 'IDENTICAL' if equal else 'DIFFERENT')
                failed += not equal
                continue
            images = []
            for tag, item in [('before', entry), ('after', other)]:
                obj = Path(item['output'])
                if not obj.is_absolute():
                    obj = Path(item['directory']) / obj
                bundle, image = work / (tag + '.bundle'), work / (tag + '.hsaco')
                try:
                    payload = section(obj, '.hip_fatbin')
                except ValueError:
                    if tag == 'before':
                        break
                    raise
                bundle.write_bytes(payload)
                command = [args.bundler, '--unbundle', '--type=o',
                           '--input=' + str(bundle), '--output=' + str(image),
                           '--targets=hip-amdgcn-amd-amdhsa--gfx1151']
                result = subprocess.run(command, capture_output=True, text=True,
                                        check=False)
                if result.returncode != 0:
                    raise SystemExit(
                        f'{source}: {tag} unbundle failed with {args.bundler}: '
                        f'{result.stderr.strip()}\nnote: unpacked objects also need '
                        f'llvm-objcopy on PATH; pass --bundler <hipClang bundle> '
                        f'or extend PATH accordingly.')
                images.append(image)
            if not images:
                print(source, 'NO-DEVICE-IMAGE (host wrapper)')
                rows -= 1
                continue
            for name in ('.text', '.rodata', '.note'):
                values = [section(image, name) for image in images]
                equal = values[0] == values[1] and bool(values[0])
                failed += not equal
                print(source, name, 'IDENTICAL' if equal else 'DIFFERENT',
                      *[hashlib.sha256(value).hexdigest() for value in values])
    print(f'{args.mode}: compared={rows} failures={failed}')
    return int(failed != 0 or rows == 0)


if __name__ == '__main__':
    raise SystemExit(main())
