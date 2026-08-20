{
  description = "strix-halo.cpp - Strix Engine for AMD Strix Halo (gfx1151 GPU + XDNA2 NPU)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      # Strix Halo is a Linux x86-64-only target (see README non-goals).
      forAllSystems = nixpkgs.lib.genAttrs [ "x86_64-linux" ];
      version = self.shortRev or self.dirtyShortRev or "dirty";

      pkgs = forAllSystems (system: import nixpkgs { inherit system; });

      strixPackages = forAllSystems (
        system:
        pkgs.${system}.callPackage ./.devops/nix/scope.nix { inherit version; }
      );

      # Offline model-conversion toolchain. Python-only; never a transitive
      # dependency of the server (see docs/QUANTIZATION.md "Offline Toolchain").
      # Unified python313 + torchWithRocm: every Strix Halo box ships ROCm, so
      # the single toolchain serves CPU flows and the --device cuda
      # calibration forward alike. gfx1151 verified on this host.
      pythonTools = system:
        let pt = pkgs.${system}.python313; in
        pt.withPackages (ps: [
          ps.torchWithRocm
          ps.transformers
          ps.safetensors
          ps.huggingface-hub
          ps.numpy
          ps.scipy
          ps.zstandard
          strixPackages.${system}.hyperloom
        ]);
    in
    {
      packages = forAllSystems (
        system:
        let
          base = strixPackages.${system}.strix;
        in
        {
          # The only package we ship: ROCm/HIP compiled for gfx1151 + XRT NPU
          # shim. The driver derivations (xrt, xrt-plugin-amdxdna) are internal
          # build inputs in scope.nix and are not exposed as flake packages.
          default = base.override {
            rocmSupport = true;
            rocmGpuTargets = [ "gfx1151" ];
          };
          aie-qwen-mtp-eh-proj =
            strixPackages.${system}.aie-qwen-mtp-eh-proj;
          aie-qwen-mtp-rmsnorm =
            strixPackages.${system}.aie-qwen-mtp-rmsnorm;
          aie-smoke = strixPackages.${system}.aie-smoke;
        }
      );

      devShells = forAllSystems (
        system:
        {
          # Unified toolchain (CPU + ROCm torch): one default shell serves all
          # offline CPU flows and the --device cuda calibration forward.
          default = pkgs.${system}.mkShell {
            inputsFrom = [ self.packages.${system}.default ];
            packages = [
              (pythonTools system)
              pkgs.${system}.rocmPackages.rocprofiler-sdk
            ];
            env = {
              ROCM_PATH = "${pkgs.${system}.rocmPackages.clr}";
              STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR =
                "${strixPackages.${system}.aie-qwen-mtp-eh-proj}";
              STRIX_AIE_QWEN_MTP_EH_PROJ_ROOT =
                "${strixPackages.${system}.aie-qwen-mtp-eh-proj}";
              STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR =
                "${strixPackages.${system}.aie-qwen-mtp-rmsnorm}";
              STRIX_AIE_QWEN_MTP_RMSNORM_ROOT =
                "${strixPackages.${system}.aie-qwen-mtp-rmsnorm}";
              STRIX_AIE_SMOKE_PROGRAM_DIR = "${strixPackages.${system}.aie-smoke}";
              STRIX_AIE_SMOKE_ROOT = "${strixPackages.${system}.aie-smoke}";
              XRT_PATH = "${strixPackages.${system}.xrt}/opt/xilinx/xrt";
            };
          };

          # Isolated Python 3.12 AIE compiler shell. Keeping this separate
          # prevents MLIR-AIE's NumPy ABI from leaking into pythonTools.
          aie = pkgs.${system}.mkShell {
            packages = [
              strixPackages.${system}.aiebu
              strixPackages.${system}.llvm-aie
              strixPackages.${system}.mlir-aie
            ];
            env = {
              MLIR_AIE_INSTALL_DIR = "${strixPackages.${system}.mlir-aie}/${pkgs.${system}.python312.sitePackages}/mlir_aie";
              PEANO_INSTALL_DIR = "${strixPackages.${system}.llvm-aie}/${pkgs.${system}.python312.sitePackages}/llvm-aie";
              STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR =
                "${strixPackages.${system}.aie-qwen-mtp-eh-proj}";
              STRIX_AIE_QWEN_MTP_EH_PROJ_ROOT =
                "${strixPackages.${system}.aie-qwen-mtp-eh-proj}";
              STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR =
                "${strixPackages.${system}.aie-qwen-mtp-rmsnorm}";
              STRIX_AIE_QWEN_MTP_RMSNORM_ROOT =
                "${strixPackages.${system}.aie-qwen-mtp-rmsnorm}";
              STRIX_AIE_SMOKE_PROGRAM_DIR = "${strixPackages.${system}.aie-smoke}";
              XRT_PATH = "${strixPackages.${system}.xrt}/opt/xilinx/xrt";
            };
          };
        }
      );

      checks = forAllSystems (
        system:
        let
          pkgsSys = pkgs.${system};
          root = toString ./.;
          mkFilteredSource =
            {
              directories ? [ ],
              files ? [ ],
            }:
            pkgsSys.lib.cleanSourceWith {
              src = ./.;
              filter =
                path: _type:
                let
                  pathString = toString path;
                  relativePath = pkgsSys.lib.removePrefix "${root}/" pathString;
                in
                pathString == root
                || builtins.elem relativePath files
                || builtins.any (
                  file: pkgsSys.lib.hasPrefix "${relativePath}/" file
                ) files
                || builtins.any (
                  directory:
                  relativePath == directory
                  || pkgsSys.lib.hasPrefix "${directory}/" relativePath
                ) directories;
            };

          formatSource = mkFilteredSource {
            directories = [
              "src"
              "tests"
            ];
            files = [ ".clang-format" ];
          };

          staticAnalysisSource = mkFilteredSource {
            directories = [
              "cmake"
              "src"
            ];
            files = [
              ".clang-format"
              ".clang-tidy"
              "CMakeLists.txt"
            ];
          };

          testSource = mkFilteredSource {
            directories = [
              "cmake"
              "src"
              "tests"
            ];
            files = [
              ".clang-format"
              "CMakeLists.txt"
            ];
          };

          dependencySource = mkFilteredSource {
            files = [
              ".devops/nix/package.nix"
              "THIRD_PARTY_NOTICES.md"
              "tools/check-dependencies.py"
            ];
          };

          formatCheck = pkgsSys.runCommand "check-format" {
            nativeBuildInputs = [ pkgsSys.clang-tools pkgsSys.findutils ];
            src = formatSource;
          } ''
            cd "$src"
            find src tests -not -path "*/fixtures/*" \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) -exec clang-format --dry-run --Werror {} +
            mkdir -p $out
            echo "PASS: Formatting check clean" > $out/result.txt
          '';

          staticAnalysisCheck = pkgsSys.runCommand "check-static-analysis" {
            nativeBuildInputs = [
              pkgsSys.stdenv.cc
              pkgsSys.clang-tools
              pkgsSys.cmake
              pkgsSys.ninja
              pkgsSys.python3
              pkgsSys.findutils
            ];
            src = staticAnalysisSource;
          } ''
            export HOME=$TMPDIR
            mkdir -p build && cd build
            cmake "$src" -GNinja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTING=OFF -DENGINE_ENABLE_HIP=OFF -DENGINE_ENABLE_XRT=OFF

            python3 - "$src" <<'PY' > tidy-files.txt
            import json
            import sys
            from pathlib import Path

            source_root = Path(sys.argv[1]).resolve()
            database_path = Path("compile_commands.json")
            commands = json.loads(database_path.read_text(encoding="utf-8"))

            unique_commands = {}
            for command in commands:
                source_file = Path(command["file"]).resolve()
                unique_commands.setdefault(str(source_file), command)

            database_path.write_text(
                json.dumps(list(unique_commands.values()), indent=2),
                encoding="utf-8",
            )

            source_dir = source_root / "src"
            for source_file in sorted(Path(path) for path in unique_commands):
                if source_file.suffix == ".cpp" and source_file.is_relative_to(source_dir):
                    print(source_file)
            PY

            tidy_jobs="''${NIX_BUILD_CORES:-1}"
            if [ "$tidy_jobs" -eq 0 ] || [ "$tidy_jobs" -gt 8 ]; then
              tidy_jobs=8
            fi
            xargs -r -n 1 -P "$tidy_jobs" clang-tidy --quiet -p . < tidy-files.txt

            mkdir -p $out
            echo "PASS: clang-tidy static analysis clean" > $out/result.txt
          '';

          dependencyInventoryCheck = pkgsSys.runCommand "check-dependency-inventory" {
            nativeBuildInputs = [ pkgsSys.python3 ];
            src = dependencySource;
          } ''
            cd "$src"
            mkdir -p $out
            python3 tools/check-dependencies.py --json-report $out/dependency-inventory.json
            echo "PASS: Dependency inventory clean" > $out/result.txt
          '';

          docsCheck = pkgsSys.runCommand "check-docs" {
            nativeBuildInputs = [ pkgsSys.python3 ];
            src = self;
          } ''
            cd "$src"
            mkdir -p $out
            python3 tools/check-docs.py --root "$src"
            echo "PASS: Documentation check clean" > $out/result.txt
          '';

          testCheck = pkgsSys.runCommand "check-tests" {
            nativeBuildInputs = [
              pkgsSys.stdenv.cc
              pkgsSys.ccache
              pkgsSys.cmake
              pkgsSys.ninja
            ];
            src = testSource;
          } ''
            export HOME=$TMPDIR
            set -euo pipefail

            ccache_launcher=
            ccache_dir=/tmp/strix-ccache
            if [[ -d "$ccache_dir" && -w "$ccache_dir" ]]; then
              export CCACHE_DIR="$ccache_dir"
              export CCACHE_BASEDIR="$src"
              export CCACHE_COMPILERCHECK=content
              export CCACHE_MAXSIZE=2G
              export CCACHE_NOHASHDIR=true
              export CCACHE_UMASK=000
              export NIX_CFLAGS_COMPILE="''${NIX_CFLAGS_COMPILE:-} -fdebug-prefix-map=$src=."
              ccache_launcher=-DCMAKE_CXX_COMPILER_LAUNCHER=ccache
              ccache --zero-stats
            fi

            mkdir -p build && cd build
            cmake "$src" -GNinja $ccache_launcher -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON -DSTRIX_ENABLE_WARNINGS=ON -DSTRIX_ENABLE_SANITIZERS=OFF
            ninja
            ctest --output-on-failure

            if [[ -n "$ccache_launcher" ]]; then
              ccache --show-stats
            fi

            mkdir -p $out
            echo "PASS: CPU build and CTest suite passed" > $out/result.txt
          '';

          # Canonical PR umbrella. Nix builds these independent derivations in
          # parallel and reuses their results across nix flake check and PR runs.
          prCheck = pkgsSys.runCommand "check-pr" { } ''
            mkdir -p $out/bin
            cp "${self.packages.${system}.default}/bin/strix" $out/bin/strix
            cp "${self.packages.${system}.default}/bin/strix-server" $out/bin/strix-server

            cat "${formatCheck}/result.txt"
            cat "${staticAnalysisCheck}/result.txt"
            cat "${dependencyInventoryCheck}/result.txt"
            cat "${docsCheck}/result.txt"
            cat "${testCheck}/result.txt"

            cat <<EOF > $out/pr-summary.txt
PR Check Summary
Status: ALL GATES PASSED
Package: ${self.packages.${system}.default.name}
Revision: ${version}
System: ${system}
Composed Gates:
  1. Format Validation (clang-format)
  2. Static Analysis (clang-tidy)
  3. Dependency and License Inventory (THIRD_PARTY_NOTICES.md)
  4. Documentation & Local Link Validation (check-docs.py)
  5. CPU Build and Runtime/Unit Tests (CTest)
Production Package Validation:
  - gfx1151 ROCm/HIP + XRT build
  - Installed strix-server version/help smoke
EOF
          '';
        in
        {
          format = formatCheck;
          static-analysis = staticAnalysisCheck;
          dependency-inventory = dependencyInventoryCheck;
          docs = docsCheck;
          tests = testCheck;
          pr = prCheck;
        }
      );
    };
}
