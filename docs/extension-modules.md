# Extension modules

> **Experimental** ([#1075](https://github.com/gufo-org/gufo/issues/1075)).
> The commands, the store layout and the module entry symbols may still change.

An extension can be added to an installed gufo without rebuilding it.
`gufo extension install` builds the extension as a loadable module against
the SDK that was installed with that gufo, and every later gufo command
loads it. Built-in extensions ([source extensions](extensions.md)) and
modules come from the same sources, `extension.json` and `extension.cmake`.

```sh
gufo extension install lfm3                             # reviewed registry release
gufo extension install lfm3@0.4.0                       # a specific reviewed release
gufo extension install ./lfm3-gufo-extension            # local directory
gufo extension install https://github.com/gufo-org/lfm3-gufo-extension@967f5e13ad501e4a505988f20694fed1e442ead1
gufo extension list                 # built-in, installed and registry extensions
gufo extension update               # newer reviewed releases; after upgrading gufo
gufo extension remove lfm3
```

## Trust

The [extension registry](https://github.com/gufo-org/gufo-extension) lists
releases whose exact source revision a Gufo maintainer reviewed.
`gufo extension install <name>[@<version>]` picks the newest reviewed release
(or the named one) for this platform and extension API, fetches its commit,
and refuses it unless the source tree matches the reviewed SHA-256 digest.
`GUFO_EXTENSION_REGISTRY` names another index, as a URL or a local file.

Gufo maintainers have not validated an extension installed from a directory or
a Git URL. Its CMake runs when it builds and its native code runs in every gufo
command that loads it, with your permissions. `install` prints a warning and
asks for confirmation, and needs `--yes` without a terminal. Review the source
first. A Git source must name a revision (`<url>@<commit-or-tag>`); the store
records the exact commit. A directory is copied, and the store records a
SHA-256 digest of the copy.

A reviewed extension still runs native code with your permissions: review is
a defense against malicious or careless code, not a sandbox.

## Why modules are built locally

Gufo promises source compatibility for extensions, not a binary ABI. The
extension surface is C++ that shares the standard library, HIP runtime and
allocators with gufo. A module is therefore built with exactly the toolchain,
flags and libraries of the gufo that loads it.

Each gufo has a **build identity**: a hash over the extension API headers, the
API version, compiler paths and versions, ROCm, global compile flags, the GPU
architecture, the build type and sanitizers. A module carries the identity it
was built for, and gufo loads only a module with its own identity. Modules are
stored per identity: after an upgrade, `gufo extension list` shows
`needs update`, and gufo warns and skips the extension until
`gufo extension update` builds it again from its pinned source. `update` also
moves an extension installed from the registry to its newest reviewed release
that this gufo supports.

## Requirements

`install` and `update` first check that everything the build needs exists,
and report every missing item together:

- CMake 3.27 or newer and the generator (Ninja) gufo was built with;
- the exact C, C++ and HIP compilers of the gufo build;
- the ROCm device libraries and every dependency the build profile pins;
- Git, for a Git source or a registry release.

These are recorded in the SDK (`profile.cmake`, `sdk.json`). Install gufo
where its toolchain is available:

| Gufo from | SDK and toolchain |
| --- | --- |
| A CMake source build | `<build>/sdk`, or `<prefix>/lib/gufo/sdk` after `cmake --install`, with the same system toolchain |
| Nix | `nix profile install github:gufo-org/gufo#sdk` (or `#release-sdk`): gufo with its `sdk` output, which pins the build toolchain |
| The `gufo-runtime` image | The SDK and its toolchain are included; mount a volume for the store |

The plain `default` and `release` Nix packages keep the SDK in the separate
`sdk` output, so their runtime closure has no compilers.
`GUFO_SDK_DIR` points gufo at an SDK outside its own prefix.

## Store

Installed extensions live in `$GUFO_EXTENSIONS_DIR`, else
`$XDG_DATA_HOME/gufo/extensions` (`~/.local/share/gufo/extensions`):

```text
installed.json                       name, version, source, revision/digest
<name>/source/                       the pinned source
<name>/modules/<build-identity>.so   one module per gufo build identity
```

A command loads the modules when it first creates the extension registry.
A running server keeps the modules it loaded. `remove` deletes the files, and
running processes keep their copy until they exit. Modules are never unloaded
from a process: device kernels, sessions and caches may still use their code.

## Conflicts

`install` refuses an extension whose name, provider ID, or artifact key or
model ID for the same capability is already provided by a built-in or another
installed extension. `--replace-builtin` installs a module in place of the
built-in extension with the same name, for example to test a newer version.
The executable then skips that built-in. At load time, a module that became
stale or conflicting, for example after an upgrade made the extension
built-in, is skipped with a warning. A broken module never prevents
`gufo extension` from repairing the store.

## Building a module from a checkout

The same build works directly in a gufo build tree:

```sh
cmake --preset release -DGUFO_EXTENSION_DIRS=/abs/my-extension \
  -DGUFO_EXTENSION_MODULES=my_extension
cmake --build --preset release --target gufo gufo_ext_module_my_extension
```

`GUFO_EXTENSION_MODULES` builds the listed extensions as
`extension-modules/gufo-ext-<name>.so` instead of linking them into gufo. A
module may use only the public `<gufo/ext/...>` surface. Its libraries take
only the headers of `gufo_ext_runtime`, `gufo_ext_core`, `gufo_ext_cache` and
`gufo_ext_hip`; the services resolve from the gufo executable, which exports
Gufo's own symbols. Modules support standalone extensions: no `depends`, and
not `kind: library`.

## Nix

A flake can pin extensions and build them against one gufo package:

```nix
{
  inputs.gufo.url = "github:gufo-org/gufo/<revision>";
  inputs.lfm3 = {
    url = "github:gufo-org/lfm3-gufo-extension/<revision>";
    flake = false;
  };
  outputs = { gufo, lfm3, ... }: {
    packages.x86_64-linux.default = gufo.lib.x86_64-linux.withExtensionModules {
      extensions = [ lfm3 ];
      # gufo = gufo.packages.x86_64-linux.release;
    };
  };
}
```

The result is gufo with a read-only store of those modules
(`GUFO_EXTENSIONS_DIR`), built in the sandbox by `gufo extension install`.

## Platforms

Linux is the production platform. On Windows, `gufo.exe` exports Gufo's
symbols through a generated `.def` file and the SDK ships its import library
(`lib/gufo.lib`); modules are DLLs linked against it. A Windows module imports
from `gufo.exe` by name, so modules load only into `gufo.exe`, not into the
`gufo-server.exe` copy. Exported data symbols need `__declspec(dllimport)` to
be referenced from a module, so an extension that uses such data does not link
on Windows yet.
