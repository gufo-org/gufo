#!/usr/bin/env python3
"""Check that Cargo, the static library, and CMake expose every Rust component."""
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def check(root):
    workspace = tomllib.loads((root / 'Cargo.toml').read_text())['workspace']['members']
    discovered = {p.parent.relative_to(root).as_posix()
                  for p in (root / 'src/core').glob('*/Cargo.toml')}
    errors = []
    if set(workspace) != discovered:
        errors.append(f'workspace members differ from component manifests: {set(workspace) ^ discovered}')
    crates = {tomllib.loads((root / p / 'Cargo.toml').read_text())['package']['name']: p
              for p in workspace}
    manifests = {name: tomllib.loads((root / p / 'Cargo.toml').read_text())
                 for name, p in crates.items()}
    linked = set()

    def visit(name):
        if name in linked:
            return
        linked.add(name)
        for dependency in manifests[name].get('dependencies', {}):
            if dependency in crates:
                visit(dependency)

    visit('gufo-rust')
    if linked != set(crates):
        errors.append(f'crates not linked through gufo-rust: {set(crates) - linked}')
    direct = {name.replace('-', '_') for name in manifests['gufo-rust']['dependencies'] if name in crates}
    externs = set(re.findall(r'^extern crate (\w+);', (root / 'src/core/rust/src/lib.rs').read_text(), re.M))
    if direct != externs:
        errors.append(f'gufo-rust dependencies and extern crates differ: {direct ^ externs}')
    expected = {p + '/include' for p in workspace if (root / p / 'include').is_dir()}
    cmake = (root / 'src/core/rust/CMakeLists.txt').read_text()
    includes = set(re.findall(r'\$\{PROJECT_SOURCE_DIR\}/(src/core/[^;" ]+/include)', cmake))
    if includes != expected:
        errors.append(f'CMake include directories differ from component headers: {includes ^ expected}')
    return errors


if __name__ == '__main__':
    errors = check(ROOT)
    if errors:
        raise SystemExit('\n'.join(errors))
    print('PASS: Rust workspace, static library and CMake registrations agree')
