{
  lib,
  stdenv,
  fetchFromGitHub,
  rocmUpdateScript,
  cmake,
  pkg-config,
  libdrm,
  libnl,
  libmnl,
  python,
  wrapPython,
  autoPatchelfHook,
}:

let
  esmi_ib_src = fetchFromGitHub {
    owner = "amd";
    repo = "esmi_ib_library";
    # ROCm 10 pins this exact commit in projects/amdsmi/CMakeLists.txt
    rev = "d494a3194ceb4cc4dbb2debf9fcbe8773c6d3bef";
    hash = "sha256-StSYzyIujaH8UJIRpfJC0lf+oVT9fWflJgQONvRqu70=";
  };
in
stdenv.mkDerivation (finalAttrs: {
  pname = "amdsmi";
  version = "10.0.0";
  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "rocm-systems";
    rev = "therock-10.0";
    sparseCheckout = [
      "projects/amdsmi"
      "shared"
    ];
    hash = "sha256-yhgSwRk9vL40bmnAE3im7DOXy6uMA7NTZxPp5fQC7Qo=";
  };
  sourceRoot = "${finalAttrs.src.name}/projects/amdsmi";

  postPatch = ''
    # ROCm 10 dropped the raw -L link line from the goamdsmi shim, so only the
    # target rename is still needed here.
    substituteInPlace goamdsmi_shim/CMakeLists.txt \
      --replace-fail "amd_smi)" ${"'"}''${AMD_SMI_TARGET})'
    # ROCm 10 replaced the tag comparison with an offline-safe check: it only
    # fetches when esmi_ib_library/src/e_smi.c is absent, which the copy below
    # satisfies.

    sed -i '/^struct drm_color_ctm_3x4 {/,/^};/d' include/libdrm/drm_mode.h

    # Manually unpack esmi_ib_src and add amd_hsmp.h so execute-process git clone doesn't run
    cp -rf --no-preserve=mode ${esmi_ib_src} ./esmi_ib_library
    mkdir -p ./esmi_ib_library/include/asm
    cp ./include/amd_smi/impl/amd_hsmp.h ./esmi_ib_library/include/asm/amd_hsmp.h
  '';

  # ROCm 10 moved the vendored DRM headers from include/amd_smi/impl/amdgpu_drm.h
  # to include/libdrm/, so amdsmi#165 no longer applies. The fix is still needed:
  # drm_color_ctm_3x4 is defined by current system libdrm, and the vendored copy
  # is unused and not part of the amdsmi public interface.
  patches = [ ];

  nativeBuildInputs = [
    cmake
    pkg-config
    wrapPython
    autoPatchelfHook
  ];

  buildInputs = [
    libdrm
    # ROCm 10's amdsmi uses system netlink via pkg-config (libnl-3.0, libmnl)
    libnl
    libmnl
  ];

  cmakeFlags = [
    # Manually define CMAKE_INSTALL_<DIR>
    # See: https://github.com/NixOS/nixpkgs/pull/197838
    "-DCMAKE_INSTALL_BINDIR=bin"
    "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DCMAKE_INSTALL_INCLUDEDIR=include"
  ];

  postInstall = ''
    mkdir -p $out/${python.sitePackages}
    ln -s $out/share/amd_smi/amdsmi $out/${python.sitePackages}/amdsmi

    makeWrapperArgs=(--prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath [ libdrm ]})
    wrapPythonProgramsIn $out
    rm $out/bin/amd-smi
    ln -sf $out/libexec/amdsmi_cli/amdsmi_cli.py $out/bin/amd-smi
  '';

  passthru.updateScript = rocmUpdateScript { inherit finalAttrs; };

  meta = {
    description = "System management interface for AMD GPUs supported by ROCm";
    homepage = "https://github.com/ROCm/rocm-systems/tree/develop/projects/amdsmi";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ lovesegfault ];
    teams = [ lib.teams.rocm ];
    platforms = [ "x86_64-linux" ];
    mainProgram = "amd-smi";
  };
})
