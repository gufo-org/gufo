{
  lib,
  stdenv,
  fetchFromGitHub,
  fetchpatch,
  rocmUpdateScript,
  pkg-config,
  cmake,
  xxd,
  python3,
  rocm-device-libs,
  rocprofiler-register,
  elfutils,
  libdrm,
  numactl,
  llvm,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "rocm-runtime";
  version = "10.0.0";

  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "rocm-systems";
    rev = "therock-10.0";
    sparseCheckout = [
      "projects/rocr-runtime"
      "shared"
    ];
    hash = "sha256-KVH8jhtjUIRNCF3gid7ZuW+XmSLRJqZtLMY+hOtXggw=";
  };
  sourceRoot = "${finalAttrs.src.name}/projects/rocr-runtime";

  cmakeBuildType = "RelWithDebInfo";
  separateDebugInfo = true;
  __structuredAttrs = true;
  strictDeps = true;

  nativeBuildInputs = [
    pkg-config
    cmake
    xxd # used by create_hsaco_ascii_file.sh
    # ROCm 10's trap_handler/CMakeLists.txt generates bytecode streams with a
    # Python script, so the interpreter has to be on the build path.
    python3
    llvm.rocm-toolchain
  ];

  buildInputs = [
    llvm.clang-unwrapped
    llvm.llvm
    elfutils
    libdrm
    numactl
    rocprofiler-register
  ];

  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=ON"
    "-DCMAKE_INSTALL_BINDIR=bin"
    "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DCMAKE_INSTALL_INCLUDEDIR=include"
  ];

  patches = [
    # The queue-allocation segfault fix (rocm-systems#2850) and the extended HIP
    # ISA compatibility check are both already in therock-10.0; re-applying them
    # to amd_gpu_agent.cpp is detected as reversed.
    # The 1 << 31 UB fix is already in therock-10.0: kfd_ioctl.h uses 1U << 31.
    # This causes a circular dependency, aqlprofile relies on hsa-runtime64
    # which is part of rocm-runtime
    # Worked around by having rocprofiler load aqlprofile directly
    ./remove-hsa-aqlprofile-dep.patch
  ];

  postPatch = ''
    patchShebangs --build \
      runtime/hsa-runtime/core/runtime/trap_handler/create_trap_handler_header.sh \
      runtime/hsa-runtime/core/runtime/blit_shaders/create_blit_shader_header.sh \
      runtime/hsa-runtime/image/blit_src/create_hsaco_ascii_file.sh
    patchShebangs --host image core runtime

    substituteInPlace CMakeLists.txt \
      --replace 'hsa/include/hsa' 'include/hsa'

    substituteInPlace runtime/hsa-runtime/image/blit_src/CMakeLists.txt \
      --replace-fail 'COMMAND clang' "COMMAND ${llvm.rocm-toolchain}/bin/clang"

    export HIP_DEVICE_LIB_PATH="${rocm-device-libs}/amdgcn/bitcode"
  '';

  passthru.updateScript = rocmUpdateScript { inherit finalAttrs; };

  meta = {
    description = "Platform runtime for ROCm";
    homepage = "https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-runtime";
    license = lib.licenses.ncsa;
    maintainers = with lib.maintainers; [ lovesegfault ];
    teams = [ lib.teams.rocm ];
    platforms = lib.platforms.linux;
  };
})
