#!/usr/bin/env python3
"""Reject Windows bundles that rely on non-system DLLs outside the bundle.

Requires pefile==2024.8.26. Checks normal and delay imports in every top-level
EXE/DLL, including SDK libraries that Gufo loads indirectly. Never resolves DLLs
against the inspection host's PATH or System32.
"""

import argparse
from pathlib import Path

import pefile

# Windows 11 inbox libraries, not development tools, CRT or GPU SDK runtimes.
SYSTEM_DLLS = frozenset(
    """advapi32.dll bcrypt.dll cfgmgr32.dll crypt32.dll gdi32.dll iphlpapi.dll
    kernel32.dll ntdll.dll ole32.dll oleaut32.dll psapi.dll secur32.dll setupapi.dll
    shell32.dll shlwapi.dll user32.dll version.dll ws2_32.dll""".split()
)


def is_system_dll(name):
    name = name.lower()
    return name in SYSTEM_DLLS or name.startswith(("api-ms-win-", "ext-ms-win-"))


def imported_dlls(path):
    with pefile.PE(str(path), fast_load=True) as pe:
        if pe.FILE_HEADER.Machine != 0x8664:
            raise ValueError(f"{path.name}: expected an x86-64 PE image")
        pe.parse_data_directories(directories=[1, 13])  # import, delay import
        return {
            entry.dll.decode("ascii").lower()
            for directory in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT")
            for entry in getattr(pe, directory, [])
        }


def check_bundle(directory):
    images = {
        path.name.lower(): path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in (".exe", ".dll")
    }
    if "gufo.exe" not in images:
        raise ValueError("bundle is missing gufo.exe")
    missing = []
    for name, path in sorted(images.items()):
        for dependency in sorted(imported_dlls(path)):
            if dependency not in images and not is_system_dll(dependency):
                missing.append(f"{name} requires unbundled {dependency}")
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        missing = check_bundle(args.directory)
    except (OSError, ValueError, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
    if missing:
        parser.exit(1, "\n".join(missing) + "\n")
    print("All imported non-system DLLs are bundled (including delay imports).")


if __name__ == "__main__":
    main()
