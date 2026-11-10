# Install

Gufo runs on Linux x86-64 with an AMD Strix Halo GPU (`gfx1151`). This page
covers the container, the Debian package, Nix, a build from source and the
Windows ZIP.

## Prepare the host

Gufo needs a recent kernel and firmware, and access to the GPU devices:

```sh
sudo usermod -aG video,render "$USER"   # then log out and back in
ls -l /dev/kfd /dev/dri/renderD*
```

As of November 2026, use Linux 6.18.4 or newer and linux-firmware 20260110 or
newer. The [host configuration guide](https://github.com/gufo-org/toolboxes/blob/main/docs/host-configuration.md)
lists the kernel parameters and known bad versions. After installing, run
`gufo diagnose` to check the setup; [troubleshooting](troubleshooting.md)
explains its findings.

## Container

The `gufo-runtime` image holds Gufo and everything it needs. Use `latest` for
the newest release, `X.Y.Z` to pin one, or `edge` to follow development.

```sh
podman pull ghcr.io/gufo-org/toolboxes/gufo-runtime:latest
podman run --rm \
  --userns=keep-id:uid=1000,gid=1000 \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --ulimit memlock=-1 \
  -p 8080:8080 -v ./models:/models:ro \
  -v gufo-extensions:/home/gufo/.local/share/gufo/extensions \
  ghcr.io/gufo-org/toolboxes/gufo-runtime:latest \
  gufo serve --host 0.0.0.0 --port 8080 llm --model /models/model.gguf
```

- Rootless Podman needs the `crun` runtime for `--group-add keep-groups`;
  check with `podman info --format '{{.Host.OCIRuntime.Name}}'`.
- Inside a container, Gufo must listen on `0.0.0.0`, or the published port
  won't answer.
- The `gufo-extensions` volume keeps [extension modules](extensions.md#modules)
  you install with `gufo extension install` across container restarts.
- On Fedora and other SELinux hosts, a misleading "Memory critical" error from
  ROCm can mean SELinux blocked `/dev/kfd`; see
  [troubleshooting](troubleshooting.md#the-gpu-isnt-found).

## Debian package

Each release attaches `gufo_<version>-1_amd64.deb` to its
[GitHub release](https://github.com/gufo-org/gufo/releases), built for Debian
13 (trixie) with AMD's ROCm 10.1 packages:

```sh
sudo apt install ./gufo_<version>-1_amd64.deb
gufo diagnose
```

## Nix

```sh
nix profile install github:gufo-org/gufo#release
gufo diagnose
```

`#release` is the shipped build with the official extensions; the default
package also builds every extension it finds. Prebuilt packages come from the
`gufo` Cachix cache. Nix asks trusted users to accept it; for other users add
it system-wide:

```text
extra-substituters = https://gufo.cachix.org
extra-trusted-public-keys = gufo.cachix.org-1:rfZe3n4lHINgyoACKzs5EthIZmBNvEyCCtufUpth0PE=
```

## Build from source

From a checkout, `nix build` gives `./result/bin/gufo` with the pinned
toolchain: GCC 16.2 and ROCm 10.1.0. Without Nix, install these first:

| Dependency | Used for |
| --- | --- |
| C++20 compiler, CMake 3.21+, Ninja, pkg-config | Building |
| ROCm HIP, hipBLAS, hipBLASLt, rocBLAS, hipCUB, rocPRIM, rocWMMA | GPU execution |
| ICU, libcurl, OpenSSL, libpng, libjpeg, libwebp | Tokenization, HTTPS, hashing, images |
| Rust 1.88 or newer, libopus | PNG, WAV and audio decoding |
| FFmpeg libraries | EmbeddingGemma 2 only |
| `ffmpeg` and `ffprobe` | Video and audio output |

On Debian or Ubuntu, with
[AMD's ROCm repository](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
set up:

```sh
sudo apt install build-essential cmake ninja-build pkg-config \
  libicu-dev libcurl4-openssl-dev libssl-dev libpng-dev libjpeg-dev libwebp-dev \
  libopus-dev \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev ffmpeg \
  hipblas-dev hipblaslt-dev rocblas-dev \
  hipcub-dev rocprim-dev rocwmma-dev
cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
cmake --install build/release
```

- Add `-DGUFO_EXTENSIONS_FROM_LOCK=ON` to build only the official extensions,
  as releases do, or `-DGUFO_EXTENSIONS=<list>` to choose; each extension
  needs only its own libraries.
- If CMake can't find ROCm, pass `-DCMAKE_PREFIX_PATH=/opt/rocm`, and
  `-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++` if it picks a system Clang.
- The Nix build applies a [ROCm runtime patch](../.devops/nix/patches/rocr-fixed-cwsr-rocm10.patch)
  that keeps GPU queues resident under memory pressure on Strix Halo. Apply it
  to your ROCm for the same behavior.

Different compilers or math libraries can change results slightly; Gufo's
published numbers use the Nix toolchain.

## Windows

Windows x86-64 is a development target. Each release attaches
`gufo-vX.Y.Z-windows-x86_64.zip`. Extract the whole ZIP and keep `gufo.exe`
next to its DLLs and kernel folders. To build it yourself, see
[building on Windows](../BUILD-WINDOWS.md).

## Model files

Gufo doesn't download models. Each [model page](models/README.md) lists the
files and pinned revisions to fetch with `hf download`.
