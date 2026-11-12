# Extensions

An extension adds a model to Gufo: its code, GPU kernels, tests and a
manifest, in one directory. This page shows how to add extensions to your Gufo
and what to check before you trust one.

## Official and community extensions

Every model in Gufo is an extension. The **official extensions** are the ones
listed in Gufo's release lock, `extensions.lock`: the maintainers review and
test them, and release packages, containers and the Windows ZIP include only
them. Every other extension is a **community extension**. It lives in its
author's repository and needs no change to Gufo.

The [extension catalogue](https://gufo.blog/extensions) lists official and
community extensions. To see what your Gufo contains:

```sh
gufo diagnose --section extensions
gufo extension list
```

> **Warning:** An extension is native code that runs with your permissions,
> and its build scripts run when it builds. Add only extensions whose source
> you've read or whose author you trust, and pin each one to an exact revision.

There are two ways to add one: install it as a module into the Gufo you
already have, or compile it into your own Gufo build.

## Modules

`gufo extension` builds an extension as a loadable **module** for your
installed Gufo. Every later `gufo` command loads it.

```sh
gufo extension install lfm3                    # newest reviewed release
gufo extension install lfm3@0.4.0              # a specific reviewed release
gufo extension install ./lfm3-gufo-extension   # a local directory
gufo extension install https://github.com/gufo-org/lfm3-gufo-extension@967f5e13ad501e4a505988f20694fed1e442ead1
gufo extension update                          # after upgrading Gufo
gufo extension remove lfm3
```

As of November 2026, modules are experimental: the commands and the store
layout may still change.

**Reviewed releases.** A name, with or without `@version`, comes from the
[extension registry](https://github.com/gufo-org/gufo-extension), which lists
releases whose exact source a Gufo maintainer reviewed. Gufo refuses the
download unless it matches the reviewed digest. `GUFO_EXTENSION_REGISTRY`
points to another registry, as a URL or a file. Review protects against
malicious or careless code; it isn't a sandbox.

**Other sources.** No maintainer has looked at a directory or Git URL, so
`install` warns and asks for confirmation; pass `--yes` when there's no
terminal. A Git URL must name a commit or tag.

**Upgrades.** A module works only with the exact Gufo build it was compiled
for: the same compiler, ROCm, flags and API. After you upgrade Gufo,
`gufo extension list` shows `needs update`, and Gufo skips the module until
`gufo extension update` rebuilds it from its pinned source. `update` also moves
registry extensions to their newest reviewed release that your Gufo supports.

**Conflicts.** `install` refuses an extension whose name, provider, files or
model ID clash with one Gufo already has. `--replace-builtin` replaces the
built-in extension of the same name, for example to test a newer version.

### What a module build needs

`install` and `update` check these first and report everything missing at
once:

- CMake 3.27 or newer, and Ninja;
- the same C, C++ and HIP compilers that built your Gufo;
- the ROCm libraries and other dependencies of that build;
- Git, for Git sources and registry releases.

| Gufo from | Toolchain |
| --- | --- |
| The `gufo-runtime` container | Included; mount a volume at `/home/gufo/.local/share/gufo/extensions` to keep the store, as in [install](install.md#container). |
| Nix | `nix profile install github:gufo-org/gufo#sdk` (or `#release-sdk`) installs Gufo with its build toolchain. |
| A source build | The build tree's `sdk`, or `<prefix>/lib/gufo/sdk` after installing, with the same system toolchain. |

`GUFO_SDK_DIR` points Gufo at an SDK elsewhere.

### Where modules live

Modules are stored in `$GUFO_EXTENSIONS_DIR`, or by default
`~/.local/share/gufo/extensions`. Removing an extension deletes its files;
processes already running keep using it until they exit.

### Pinning modules with Nix

A flake can build a Gufo with a fixed set of modules:

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
    };
  };
}
```

## Compiling an extension in

When you build Gufo yourself, it compiles every extension in its tree plus the
directories you add:

```sh
cmake --preset release -DGUFO_EXTENSION_DIRS=/absolute/path/to/lfm3-gufo-extension
cmake --build --preset release --parallel 4
```

With Nix, override the package:

```nix
packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
  extensions = [ lfm3 ];
};
```

The build rejects a community extension that claims a file type, option or
model ID that an official extension owns, before compiling anything.

## Platforms

Linux on Strix Halo is the production platform. On Windows, modules are DLLs
that load only into `gufo.exe`. Each extension's manifest says which platforms
it supports.

To write your own, see [writing an extension](writing-extensions.md).
