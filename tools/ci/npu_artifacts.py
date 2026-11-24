#!/usr/bin/env python3
"""Validate the CMake-declared NPU runtime files in an installed bundle."""

import argparse
from pathlib import Path, PurePosixPath


def read_manifest(path):
    names = path.read_text(encoding='utf-8').splitlines()
    if not names or len(names) != len(set(names)):
        raise ValueError(f'Empty or duplicate NPU artifact manifest: {path}')
    for name in names:
        parts = PurePosixPath(name).parts
        if (len(parts) != 2 or any(part in ('.', '..') for part in parts)
                or '\\' in name or ':' in name or PurePosixPath(name).is_absolute()):
            raise ValueError(f'Invalid NPU artifact path: {name}')
    return names


def check_artifacts(root, names):
    for name in names:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing or empty NPU artifact: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    try:
        names = read_manifest(args.manifest)
        check_artifacts(args.root, names)
        if read_manifest(args.root / 'npu-artifacts.txt') != names:
            raise ValueError('Bundled NPU manifest differs from the build')
    except (OSError, ValueError) as error:
        parser.exit(1, f'{error}\n')
    print(f'Validated {len(names)} NPU runtime artifacts.')


if __name__ == '__main__':
    main()
