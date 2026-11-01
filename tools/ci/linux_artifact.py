#!/usr/bin/env python3
"""Package Linux binaries with a relocatable gfx1151 runtime; Nix is build-host only."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


STORE = Path('/nix/store')


def output(*args):
    return subprocess.check_output(args, text=True).strip()


def runtime_path(source, root):
    return root / 'runtime' / source.relative_to(STORE)


def runtime_file(path):
    architectures = re.findall(r'gfx[0-9a-f]+', path.name)
    if architectures and any(arch != 'gfx1151' for arch in architectures):
        return False
    return ('.so' in path.name or 'rocblas/library' in path.as_posix()
            or 'hipblaslt/library' in path.as_posix())


def copy_runtime(source, root):
    destination = runtime_path(source, root)
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        target = runtime_path(source.resolve(strict=True), root)
        destination.symlink_to(os.path.relpath(target, destination.parent))
    else:
        shutil.copyfile(source, destination)
        destination.chmod(source.stat().st_mode & 0o777)


def copy_share(source, destination):
    shutil.copytree(source, destination, symlinks=False)
    # Nix store directories are read-only. Persistent self-hosted workspaces
    # must be able to delete the unpacked copy before the next run.
    for directory in [destination, *destination.rglob('*')]:
        if directory.is_dir():
            directory.chmod(directory.stat().st_mode | 0o700)


def launcher(directory, name, loader, libraries, environment):
    script = directory / name
    # Use shell builtins only before invoking the bundled ELF loader. The
    # loader's --library-path applies to Gufo and dlopen, without leaking a
    # newer libc through LD_LIBRARY_PATH into host programs spawned by Gufo.
    text = '#!/bin/sh\nset -eu\n'
    text += 'here=$(CDPATH= cd -- "${0%/*}" && pwd -P)\n'
    text += 'root=$(CDPATH= cd -- "$here/.." && pwd -P)\n'
    text += 'unset LD_LIBRARY_PATH\n'
    for key, value in environment.items():
        text += f'export {key}="$root/{value}"\n'
    paths = ':'.join('$root/' + path for path in libraries)
    text += f'exec "$root/{loader}" --library-path "{paths}" "$here/libexec/{name}" "$@"\n'
    script.write_text(text)
    script.chmod(0o755)


def pack(package, revision, destination):
    package = package.resolve(strict=True)
    version = output(str(package / 'bin/gufo'), '--version')
    if not version.splitlines()[0].endswith(f'({revision[:7]})'):
        raise ValueError(f'Package revision mismatch: {version}; expected {revision}')
    root = destination.parent
    closure = [Path(path) for path in output('nix-store', '--query', '--requisites', str(package)).splitlines()]
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'libexec').mkdir()
    executable = package / 'bin/.gufo-wrapped'
    if not executable.is_file():
        executable = package / 'bin/gufo'
    if executable.read_bytes()[:4] != b'\x7fELF':
        raise ValueError('Production package does not contain an ELF executable')
    shutil.copy2(executable, destination / 'libexec/gufo')
    copy_share(package / 'share', destination / 'share')
    for dependency in closure:
        if dependency == package:
            continue
        for library in (dependency / 'lib').rglob('*'):
            if (library.is_file() or library.is_symlink()) and runtime_file(library):
                copy_runtime(library, root)
    # Preserve the production package's patched ROCr precedence, then use only
    # the pinned closure's libraries. No libraries come from the GPU host.
    wrapper = (package / 'bin/gufo').read_text()
    patched = re.search(r"LD_LIBRARY_PATH='(/nix/store/[^']+)'", wrapper)
    if not patched:
        raise ValueError('Production package does not select its patched ROCr runtime')
    preferred = runtime_path(Path(patched[1]), root).relative_to(root).as_posix()
    directories = sorted({path.parent.relative_to(root).as_posix()
                          for path in (root / 'runtime').rglob('*')
                          if path.is_file() and '.so' in path.name
                          and any(path.is_relative_to(runtime_path(dep, root)) for dep in closure)})
    libraries = [preferred] + [path for path in directories if path != preferred]
    loaders = sorted({(dep / 'lib/ld-linux-x86-64.so.2').resolve() for dep in closure
                      if (dep / 'lib/ld-linux-x86-64.so.2').is_file()})
    if len(loaders) != 1:
        raise ValueError('Expected one pinned glibc ELF loader')
    loader = runtime_path(loaders[0], root).relative_to(root).as_posix()
    environment = {}
    for dependency in closure:
        for key, suffix in [('HIPBLASLT_TENSILE_LIBPATH', 'lib/hipblaslt/library'),
                            ('ROCBLAS_TENSILE_LIBPATH', 'lib/rocblas/library')]:
            if (dependency / suffix).is_dir():
                environment[key] = runtime_path(dependency / suffix, root).relative_to(root).as_posix()
        ids = dependency / 'share/libdrm/amdgpu.ids'
        if ids.is_file():
            copy_runtime(ids, root)
            environment['AMDGPU_ASIC_ID_TABLE_PATHS'] = runtime_path(ids.parent, root).relative_to(root).as_posix()
    ffmpeg = [dep for dep in closure if (dep / 'bin/ffmpeg').is_file()]
    if len(ffmpeg) != 1:
        raise ValueError('Expected one pinned FFmpeg package')
    for name in ('ffmpeg', 'ffprobe'):
        shutil.copy2(ffmpeg[0] / 'bin' / name, destination / 'libexec' / name)
        launcher(destination, name, loader, libraries, {})
        environment['GUFO_' + name.upper()] = destination.relative_to(root).as_posix() + '/' + name
    launcher(destination, 'gufo', loader, libraries, environment)
    (destination / 'gufo-server').symlink_to('gufo')
    executable = destination / 'libexec/gufo'
    with executable.open('rb') as file:
        digest = hashlib.file_digest(file, 'sha256').hexdigest()
    # Harnesses hash the executable entry point; retain the actual ELF identity
    # in that entry point even when only the compiled binary changes.
    script = destination / 'gufo'
    script.write_text(script.read_text().replace('set -eu\n', f'# Gufo ELF SHA-256: {digest}\nset -eu\n', 1))
    manifest = {'schema': 2, 'revision': revision, 'version': version, 'sha256': digest,
                'loader': loader, 'libraries': libraries}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{version}: portable executable and gfx1151 runtime prepared', flush=True)


def verify(source, revision):
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['schema'] != 2 or manifest['revision'] != revision:
        raise ValueError('Artifact does not match the authorized revision')
    with (source / 'libexec/gufo').open('rb') as file:
        if hashlib.file_digest(file, 'sha256').hexdigest() != manifest['sha256']:
            raise ValueError('Linux binary checksum mismatch')
    version = output(str(source / 'gufo'), '--version')
    if version != manifest['version'] or not version.splitlines()[0].endswith(f'({revision[:7]})'):
        raise ValueError('Linux binary has the wrong revision')
    root = source.parent
    linked = output(str(root / manifest['loader']), '--library-path',
                    ':'.join(str(root / path) for path in manifest['libraries']),
                    '--list', str(source / 'libexec/gufo'))
    if '=> /nix/store/' in linked or 'not found' in linked:
        raise ValueError('Linux binary depends on an unbundled runtime library')
    print(version, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('pack')
    create.add_argument('--package', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    create.add_argument('--revision', required=True)
    check = commands.add_parser('verify')
    check.add_argument('--source', type=Path, required=True)
    check.add_argument('--revision', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        parser.error('revision must be an immutable full Git commit SHA')
    if args.command == 'pack':
        pack(args.package, args.revision, args.output)
    else:
        verify(args.source.resolve(), args.revision)


if __name__ == '__main__':
    main()
