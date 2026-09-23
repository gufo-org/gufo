{
  lib,
  stdenv,
  fetchFromGitHub,
  rocmPackages,
}:

let
  revision = "0aaea5a238fb41a35106a551e73c8409dfb751ac";
  headers = with rocmPackages; [
    hipblas
    hipblas-common
    hipblaslt
    rocblas
    rocwmma
    hipcub
    rocprim
  ];
  libraries = with rocmPackages; [
    hipblas
    hipblaslt
    rocblas
  ];
in
stdenv.mkDerivation (finalAttrs: {
  pname = "ds4-reference";
  version = "0-unstable-2026-09-20";

  src = fetchFromGitHub {
    owner = "antirez";
    repo = "ds4";
    rev = revision;
    hash = "sha256-Bo/td1HwVjw6bwz3BDwTP+ZSudkVFCo2aIVK4axvmXg=";
  };

  nativeBuildInputs = [ rocmPackages.clr ];
  buildInputs = [ rocmPackages.clr ] ++ headers;
  dontConfigure = true;

  # Upstream's Strix Halo build, with explicit Nix paths and a reproducible
  # CPU target. No inference patches or build-time GPU detection.
  makeFlags = [
    "HIPCC=${rocmPackages.clr}/bin/hipcc"
    "ROCM_ARCH=gfx1151"
    "NATIVE_CPU_FLAG=-march=znver5"
    (
      "ROCM_CFLAGS=-O3 -ffast-math -g -fno-finite-math-only -pthread"
      + " -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument"
      + " --rocm-path=${rocmPackages.clr} --offload-arch=gfx1151"
      + lib.concatMapStrings (p: " -I${lib.getDev p}/include") headers
    )
    (
      "ROCM_LDLIBS=-lm -pthread"
      + lib.concatMapStrings (p: " -L${lib.getLib p}/lib") libraries
      + " -lhipblas -lhipblaslt -lrocblas"
    )
  ];

  buildPhase = ''
    runHook preBuild
    make strix-halo -j"$NIX_BUILD_CORES" ${lib.escapeShellArgs finalAttrs.makeFlags}
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    install -Dm755 -t "$out/bin" ds4 ds4-server ds4-bench ds4-eval ds4-agent
    install -Dm644 LICENSE "$out/share/licenses/ds4-reference/LICENSE"
    cp -r licenses "$out/share/licenses/ds4-reference/third-party"
    printf '%s\n' '${revision}' > "$out/share/ds4-revision"
    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/ds4-server" --help >/dev/null
    "$out/bin/ds4-bench" --help >/dev/null
    runHook postInstallCheck
  '';

  passthru = { inherit revision; };

  meta = {
    description = "Optional antirez/ds4 benchmark reference for Strix Halo";
    homepage = "https://github.com/antirez/ds4";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "ds4-server";
  };
})
