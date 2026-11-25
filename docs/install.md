# Install

Gufo runs on Linux x86-64 with an AMD Strix Halo GPU (`gfx1151`). This page
covers the container, the Debian package, Nix, a build from source and the
Windows ZIP.

## Prepare the host

Gufo needs a recent kernel and firmware, and access to the GPU and NPU
devices:

```sh
sudo usermod -aG video,render "$USER"   # then log out and back in
ls -l /dev/kfd /dev/dri/renderD* /dev/accel/accel0
```

`/dev/accel/accel0` is the XDNA NPU, from the kernel's `amdxdna` driver.

As of November 2026, use Linux 6.18.4 or newer and linux-firmware 20260110 or
newer. The [host configuration guide](https://github.com/gufo-org/toolboxes/blob/main/docs/host-configuration.md)
lists known bad versions. After installing, run `gufo diagnose` to check the
setup; [troubleshooting](troubleshooting.md) explains its findings.

### Memory settings

The GPU shares system RAM. Let it use almost all of it, through the kernel's
**GTT** (the system memory the GPU can map), rather than reserving it in the
BIOS:

1. In the BIOS, set the GPU's dedicated memory (UMA frame buffer or iGPU
   memory size) to its minimum, 512 MiB.
2. Add these kernel parameters to the boot command line (on NixOS,
   `boot.kernelParams`). For 128 GiB of RAM they give the GPU 124 GiB and leave
   4 GiB to the system:

   ```text
   amdgpu.gttsize=126976 ttm.pages_limit=32505856
   ```

   For another size, `amdgpu.gttsize` is your RAM in MiB minus 4096, and
   `ttm.pages_limit` is that value times 256 (4 KiB pages).
3. Allow unlimited locked memory: `LimitMEMLOCK=infinity` for a systemd
   service, or `memlock unlimited` in `/etc/security/limits.d/` for your
   user.

Reboot and check:

```sh
cat /sys/class/drm/card*/device/mem_info_vram_total   # 536870912 (512 MiB)
cat /sys/class/drm/card*/device/mem_info_gtt_total    # about your RAM minus 4 GiB
ulimit -l                                             # unlimited
```

Gufo runs with other values, but with a GTT limit well below your RAM the RAM
cache cannot borrow memory that sessions have reserved and not used yet, and
with locked memory limited, GPU work can stall for seconds while the kernel
compacts memory.

For `--npu`, keep the NPU out of runtime suspend. Otherwise, the first request
after 5 s idle takes about 110 ms longer. The setting resets at boot:

```sh
echo on | sudo tee /sys/class/accel/accel0/device/power/control
```

The NPU power mode (`sudo xrt-smi configure --pmode performance`) is optional.
As of November 2026 it made no measurable difference to EmbeddingGemma 2.

> **Warning:** Heavy GPU and NPU work at the same time can hang some Strix
> Halo hosts until power-cycled
> ([amd/xdna-driver#1605](https://github.com/amd/xdna-driver/issues/1605)).

## Container

The `gufo-runtime` image holds Gufo and everything it needs. Use `latest` for
the newest release, `X.Y.Z` to pin one, or `edge` to follow development.

```sh
podman pull ghcr.io/gufo-org/toolboxes/gufo-runtime:latest
podman run --rm \
  --userns=keep-id:uid=1000,gid=1000 \
  --device /dev/kfd --device /dev/dri --device /dev/accel/accel0 \
  --group-add keep-groups --ulimit memlock=-1 \
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

[`apt.gufo.blog`](https://github.com/gufo-org/apt#install) serves every release
for Debian 13 (trixie), and `apt upgrade` installs new ones. With [AMD's ROCm 10.1 repository](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) set up:

```sh
curl -fsSLo /tmp/gufo-keyring.deb https://apt.gufo.blog/gufo-archive-keyring.deb
sudo apt install /tmp/gufo-keyring.deb && sudo apt update && sudo apt install gufo
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
| XRT with the amdxdna plugin | NPU execution |
| ICU, libcurl, OpenSSL, libwebp | Tokenization, HTTPS, hashing, images |
| Rust 1.88 or newer, libopus | PNG encoding; PNG, JPEG, WAV and audio decoding |
| FFmpeg libraries | EmbeddingGemma 2 only |
| CMake 3.26+ (4.2+ on Windows), Python 3.12+, network access | Building the [Loom runtime](models/loom.md#platforms) from its pinned release |
| `ffmpeg` and `ffprobe` | Video and audio output |

On Debian or Ubuntu, with
[AMD's ROCm repository](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
set up:

```sh
sudo apt install build-essential cmake ninja-build pkg-config \
  libicu-dev libcurl4-openssl-dev libssl-dev libwebp-dev \
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
- XRT has no distribution package. Build it with its plugin from
  [xdna-driver](https://github.com/amd/xdna-driver), which installs to
  `/opt/xilinx/xrt`; for another prefix, set `XILINX_XRT` to it. The Nix
  build pins the pair in [`.devops/nix/xrt`](../.devops/nix/xrt/default.nix).
  Only `lib/gufo/libgufo_npu_xrt.so`, which Gufo loads on first NPU use,
  links XRT: `gufo` starts without XRT and runs on the GPU, and libraries
  bundled in XRT's `lib` directory never replace the ones `gufo` links.
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

Extensions can pin weights by Hugging Face revision, file size and SHA-256.
`gufo model` downloads those weights and checks each file:

```sh
gufo model list                   # every model and the extension that offers it
gufo model install qwen3-asr      # download it
gufo model install loom-fun/<name>   # an extension resolves the repository
gufo transcribe --model qwen3-asr --audio speech.wav
gufo model update                 # move to the files extensions pin now
gufo model remove qwen3-asr
```

`gufo model list --registry` adds the models of [registry](extensions.md)
extensions you have not installed, read with Git from each reviewed commit
without building it. A model without a size pins no download; pass its weights'
path to `--model`. `--json` reports `extensionStatus` and `downloadable`.

Pass an installed model's ID to `--model`; a path still works for weights you
download yourself. `install` also fetches a model's default companions, such as
an image projector or a draft model, and prints their paths. `--component ROLE`
adds an optional one.

Downloads run over 16 parallel connections (`--connections`), resume after an
interruption and reuse files already in the Hugging Face cache. Models live in
`$GUFO_MODELS_DIR`, else `$XDG_DATA_HOME/gufo/models`; models that share a file
store it once.

A Hugging Face token is optional. Set `HF_TOKEN`, or run `hf auth login`, for
gated models and higher rate limits; Gufo sends it only to Hugging Face, never
to the download servers it redirects to. `HF_ENDPOINT` selects a mirror.

## Scheduled models on NixOS

Import `gufo.nixosModules.model-sync` from your Gufo flake input:

```nix
{
  imports = [ gufo.nixosModules.model-sync ];
  services.gufo-model-sync = {
    enable = true;
    models = [ "qwen27-q4" "qwen3-asr" ];
    onCalendar = "daily";
    randomizedDelaySec = "30m";
  };
}
```

Only the calendar timer starts sync; activation and boot do not start it or
catch up missed runs. It installs missing models and calls `gufo model update`
for installed ones, following the files pinned by their local extensions.
It never installs extensions, deletes undeclared models or starts Gufo.
llama-swap can launch Gufo independently using `--model <ID>`.

The default account is `gufo` and the model directory is `/var/lib/gufo/models`.
Set `user`, `group` and `directory` to share readable files with your inference
process; give that process the same `GUFO_MODELS_DIR`.

Evaluation rejects IDs that no extension built into `package` offers. To sync
an external extension's model, [compile it in](extensions.md#compiling-an-extension-in)
and pass that package as `package`. Each run still checks `gufo model list --json`
before downloading. For custom deployment hooks, `gufo.lib.mkModelSync {
inherit pkgs; models = [ "qwen27-q4" ]; }` returns the same executable without
configuring a service or timer.
