{
  description = "gufo - Gufo Engine for AMD Strix Halo (gfx1151 GPU + XDNA2 NPU)";

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

      gufoPackages = forAllSystems (
        system:
        pkgs.${system}.callPackage ./.devops/nix/scope.nix { inherit version; }
      );

      alexnetWeights = system:
        pkgs.${system}.fetchurl {
          url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth";
          hash = "sha256-e+W+eRFZRysfvzxpeW98sw3KethGbC33AFjDcRbN7gI=";
        };

      alexnetTorchHome = system:
        pkgs.${system}.runCommand "torchvision-alexnet-cache" { } ''
          mkdir -p "$out/hub/checkpoints"
          ln -s ${alexnetWeights system} \
            "$out/hub/checkpoints/alexnet-owt-7be5be79.pth"
        '';

      # Offline model-conversion toolchain. Python-only; never a transitive
      # dependency of the server (see docs/QUANTIZATION.md "Offline Toolchain").
      # Unified python313 + torchWithRocm: every Strix Halo box ships ROCm, so
      # the single toolchain serves CPU flows and the --device cuda
      # calibration forward alike. gfx1151 verified on this host.
      pythonTools = system:
        let pt = pkgs.${system}.python313; in
        pt.withPackages (
          ps:
          let
            # Nixpkgs' default torchvision/lpips closures use CPU torch.
            # Override both edges so evaluation has one ROCm torch derivation.
            torchvisionRocm = ps.torchvision.override {
              torch = ps.torchWithRocm;
            };
            lpipsRocm = ps.lpips.override {
              torch = ps.torchWithRocm;
              torchvision = torchvisionRocm;
            };
            accelerateRocm = ps.accelerate.override {
              torch = ps.torchWithRocm;
            };
          in
          [
            ps.torchWithRocm
            torchvisionRocm
            ps.transformers
            accelerateRocm
            ps.safetensors
            ps.huggingface-hub
            ps.requests
            lpipsRocm
            ps.numpy
            ps.scipy
            ps.pandas
            ps.zstandard
            gufoPackages.${system}.hyperloom
          ]
        );
    in
    {
      lib = {
        mkGufoServe =
          {
            pkgs ? null,
            system ? pkgs.system or "x86_64-linux",
            gufo ? self.packages.${system}.default,
            ...
          }@args:
          let
            targetScope = gufoPackages.${system};
            mkServeFn = targetScope.mkServe.override {
              inherit gufo;
            };
            fnArgs = builtins.removeAttrs args [ "pkgs" "system" "gufo" ];
          in
          mkServeFn fnArgs;
        mkServe = self.lib.mkGufoServe;
      } // forAllSystems (system: {
        mkGufoServe =
          args:
          self.lib.mkGufoServe (args // { inherit system; });
        mkServe =
          args:
          self.lib.mkGufoServe (args // { inherit system; });
      });

      packages = forAllSystems (
        system:
        let
          base = gufoPackages.${system}.gufo;
        in
        {
          # The only package we ship: ROCm/HIP compiled for gfx1151 + XRT NPU
          # shim. The driver derivations (xrt, xrt-plugin-amdxdna) are internal
          # build inputs in scope.nix and are not exposed as flake packages.
          default = base.override {
            rocmSupport = true;
            rocmGpuTargets = [ "gfx1151" ];
          };
          hrx = base.override {
            rocmSupport = true;
            rocmGpuTargets = [ "gfx1151" ];
            hrxSupport = true;
          };
          aie-qwen-mtp-eh-proj =
            gufoPackages.${system}.aie-qwen-mtp-eh-proj;
          aie-qwen-mtp-rmsnorm =
            gufoPackages.${system}.aie-qwen-mtp-rmsnorm;
          aie-smoke = gufoPackages.${system}.aie-smoke;
          hrx-system = gufoPackages.${system}.hrx-system;
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
              pkgs.${system}.clang-tools
              pkgs.${system}.ffmpeg-headless
              pkgs.${system}.sox
              pkgs.${system}.rocmPackages.rocprofiler-sdk
              pkgs.${system}.sqlite
            ];
            env = {
              ROCM_PATH = "${pkgs.${system}.rocmPackages.clr}";
              GUFO_HIPCUB_ROOT = "${pkgs.${system}.rocmPackages.hipcub}";
              GUFO_ROCPRIM_ROOT = "${pkgs.${system}.rocmPackages.rocprim}";
              GUFO_ROCWMMA_ROOT = "${pkgs.${system}.rocmPackages.rocwmma}";
              GUFO_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR =
                "${gufoPackages.${system}.aie-qwen-mtp-eh-proj}";
              GUFO_AIE_QWEN_MTP_EH_PROJ_ROOT =
                "${gufoPackages.${system}.aie-qwen-mtp-eh-proj}";
              GUFO_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR =
                "${gufoPackages.${system}.aie-qwen-mtp-rmsnorm}";
              GUFO_AIE_QWEN_MTP_RMSNORM_ROOT =
                "${gufoPackages.${system}.aie-qwen-mtp-rmsnorm}";
              GUFO_AIE_SMOKE_PROGRAM_DIR = "${gufoPackages.${system}.aie-smoke}";
              GUFO_AIE_SMOKE_ROOT = "${gufoPackages.${system}.aie-smoke}";
              XRT_PATH = "${gufoPackages.${system}.xrt}/opt/xilinx/xrt";
              TORCH_HOME = "${alexnetTorchHome system}";
              LD_LIBRARY_PATH = pkgs.${system}.lib.makeLibraryPath [
                pkgs.${system}.stdenv.cc.cc.lib
                pkgs.${system}.zlib
              ];
            };
          };

          # Isolated Python 3.12 AIE compiler shell. Keeping this separate
          # prevents MLIR-AIE's NumPy ABI from leaking into pythonTools.
          aie = pkgs.${system}.mkShell {
            packages = [
              gufoPackages.${system}.aiebu
              gufoPackages.${system}.llvm-aie
              gufoPackages.${system}.mlir-aie
            ];
            env = {
              MLIR_AIE_INSTALL_DIR = "${gufoPackages.${system}.mlir-aie}/${pkgs.${system}.python312.sitePackages}/mlir_aie";
              PEANO_INSTALL_DIR = "${gufoPackages.${system}.llvm-aie}/${pkgs.${system}.python312.sitePackages}/llvm-aie";
              GUFO_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR =
                "${gufoPackages.${system}.aie-qwen-mtp-eh-proj}";
              GUFO_AIE_QWEN_MTP_EH_PROJ_ROOT =
                "${gufoPackages.${system}.aie-qwen-mtp-eh-proj}";
              GUFO_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR =
                "${gufoPackages.${system}.aie-qwen-mtp-rmsnorm}";
              GUFO_AIE_QWEN_MTP_RMSNORM_ROOT =
                "${gufoPackages.${system}.aie-qwen-mtp-rmsnorm}";
              GUFO_AIE_SMOKE_PROGRAM_DIR = "${gufoPackages.${system}.aie-smoke}";
              XRT_PATH = "${gufoPackages.${system}.xrt}/opt/xilinx/xrt";
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
                  || pkgsSys.lib.hasPrefix "${relativePath}/" directory
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
              "tools/gufo"
            ];
            files = [
              ".clang-format"
              "CMakeLists.txt"
            ];
          };

          h3ManifestSource = mkFilteredSource {
            directories = [
              "tools/gufo"
            ];
            files = [
              "tests/tools/test_h3_manifest.py"
              "tools/h3/gufo-h3-manifest.py"
            ];
          };

          h3QualitySource = mkFilteredSource {
            directories = [
              "tools/gufo"
            ];
            files = [
              "src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
              "tests/fixtures/minimax_h3/quality-contract-v1.json"
              "tests/tools/h3_cache_control_test.py"
              "tests/tools/h3_denoiser_golden_test.py"
              "tests/tools/h3_latent_quality_test.py"
              "tests/tools/h3_lpips_test.py"
              "tests/tools/h3_preset_quality_test.py"
              "tests/tools/h3_profile_test.py"
              "tests/tools/h3_profile_report_test.py"
              "tests/tools/h3_rng_test.py"
              "tests/tools/test_h3_quality.py"
              "tools/h3/gufo-h3-quality.py"
            ];
          };

          dependencySource = mkFilteredSource {
            files = [
              ".devops/nix/package.nix"
              "flake.nix"
              "THIRD_PARTY_NOTICES.md"
              "tools/ci/check-dependencies.py"
            ];
          };

          formatCheck = pkgsSys.runCommand "check-format" {
            nativeBuildInputs = [ pkgsSys.clang-tools pkgsSys.findutils ];
            src = formatSource;
          } ''
            cd "$src"
            find src tests \
              -not -path "*/fixtures/*" \
              -not -path "*/vendor/*" \
              -not -path "src/models/deepseek_v4_flash/runtime/*" \
              -not -path "src/models/deepseek_v4_flash/kernels/rocm/*" \
              \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) \
              -exec clang-format --dry-run --Werror {} +
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
              pkgsSys.icu
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
            python3 tools/ci/check-dependencies.py --json-report $out/dependency-inventory.json
            echo "PASS: Dependency inventory clean" > $out/result.txt
          '';

          docsCheck = pkgsSys.runCommand "check-docs" {
            nativeBuildInputs = [ pkgsSys.python3 ];
            src = self;
          } ''
            cd "$src"
            mkdir -p $out
            python3 tools/ci/check-docs.py --root "$src"
            echo "PASS: Documentation check clean" > $out/result.txt
          '';

          h3ManifestCheck = pkgsSys.runCommand "check-h3-manifest" {
            nativeBuildInputs = [ pkgsSys.python3 ];
            src = h3ManifestSource;
          } ''
            cd "$src"
            python3 tests/tools/test_h3_manifest.py
            mkdir -p $out
            echo "PASS: MiniMax H3 manifest tests clean" > $out/result.txt
          '';

          h3QualityCheck = pkgsSys.runCommand "check-h3-quality" {
            nativeBuildInputs = [
              (pkgsSys.python3.withPackages (ps: [ ps.numpy ]))
            ];
            src = h3QualitySource;
          } ''
            cd "$src"
            python3 tests/tools/test_h3_quality.py
            python3 tests/tools/h3_rng_test.py
            python3 tests/tools/h3_cache_control_test.py
            python3 tests/tools/h3_latent_quality_test.py
            python3 tests/tools/h3_preset_quality_test.py
            python3 tests/tools/h3_profile_test.py
            python3 tests/tools/h3_profile_report_test.py
            mkdir -p $out
            echo "PASS: MiniMax H3 quality-oracle tests clean" > $out/result.txt
          '';

          h3MlQualityCheck = pkgsSys.runCommand "check-h3-ml-quality" {
            nativeBuildInputs = [ (pythonTools system) ];
            src = h3QualitySource;
            TORCH_HOME = "${alexnetTorchHome system}";
          } ''
            cd "$src"
            python3 tests/tools/h3_denoiser_golden_test.py
            python3 tests/tools/h3_lpips_test.py
            mkdir -p $out
            echo "PASS: MiniMax H3 pinned teacher and offline LPIPS clean" > $out/result.txt
          '';

          testCheck = pkgsSys.runCommand "check-tests" {
            nativeBuildInputs = [
              pkgsSys.stdenv.cc
              pkgsSys.ccache
              pkgsSys.cmake
              pkgsSys.ninja
              (pkgsSys.python3.withPackages (ps: [ ps.numpy ]))
              pkgsSys.icu
            ];
            src = testSource;
          } ''
            export HOME=$TMPDIR
            set -euo pipefail

            ccache_launcher=
            ccache_dir=/tmp/gufo-ccache
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
            cmake "$src" -GNinja $ccache_launcher -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON -DGUFO_ENABLE_WARNINGS=ON -DGUFO_ENABLE_SANITIZERS=OFF
            ninja
            ctest --output-on-failure

            if [[ -n "$ccache_launcher" ]]; then
              ccache --show-stats
            fi

            mkdir -p $out
            echo "PASS: CPU build and CTest suite passed" > $out/result.txt
          '';

          mkServeCheck =
            let
              cmd = self.lib.${system}.mkGufoServe {
                model = "/var/models/qwen.gguf";
                context = 4096;
                servedModelName = "qwen-test";
                speculative = "dflash2";
                draftModel = "/var/models/qwen-draft.gguf";
                port = 9000;
              };
            in
            pkgsSys.runCommand "check-mk-serve" { } ''
              # Verify the synthesized CLI string contains expected flags and binary path
              cmd_str="${cmd}"
              echo "$cmd_str" | grep -F "/bin/gufo serve"
              echo "$cmd_str" | grep -F -- "--model /var/models/qwen.gguf"
              echo "$cmd_str" | grep -F -- "--context 4096"
              echo "$cmd_str" | grep -F -- "--served-model-name qwen-test"
              echo "$cmd_str" | grep -F -- "--speculative dflash2"
              echo "$cmd_str" | grep -F -- "--dflash-model /var/models/qwen-draft.gguf"
              echo "$cmd_str" | grep -F -- "--port 9000"
              mkdir -p $out
              echo "PASS: mkGufoServe CLI string check passed" > $out/result.txt
            '';

          # Canonical PR umbrella. Nix builds these independent derivations in
          # parallel and reuses their results across flake checks and PR runs.
          # The ROCm PyTorch/LPIPS closure remains an explicit h3-ml-quality
          # check: realizing its multi-gigabyte offline-evaluation toolchain on
          # every hosted PR runner exhausts the runner disk before tests start.
          prCheck = pkgsSys.runCommand "check-pr" { } ''
            mkdir -p $out/bin
            cp "${self.packages.${system}.default}/bin/gufo" $out/bin/gufo
            test -x "${self.packages.${system}.hrx}/bin/gufo"

            cat "${formatCheck}/result.txt"
            cat "${staticAnalysisCheck}/result.txt"
            cat "${dependencyInventoryCheck}/result.txt"
            cat "${docsCheck}/result.txt"
            cat "${h3ManifestCheck}/result.txt"
            cat "${h3QualityCheck}/result.txt"
            cat "${testCheck}/result.txt"
            cat "${mkServeCheck}/result.txt"

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
  5. MiniMax H3 Source-Manifest Validation
  6. MiniMax H3 Quality-Oracle Validation
  7. CPU Build and Runtime/Unit Tests (CTest)
  8. Declarative Server Wrapper Generation (mkGufoServe)
Explicit Offline Gate (not in hosted PR closure):
  - MiniMax H3 Pinned Teacher & Offline LPIPS Validation
Production Package Validation:
  - gfx1151 ROCm/HIP + XRT build
  - gfx1151 HRX/Loom compile and package build
  - Installed gufo version/help smoke
EOF
          '';
        in
        {
          format = formatCheck;
          static-analysis = staticAnalysisCheck;
          dependency-inventory = dependencyInventoryCheck;
          docs = docsCheck;
          h3-manifest = h3ManifestCheck;
          h3-quality = h3QualityCheck;
          h3-ml-quality = h3MlQualityCheck;
          tests = testCheck;
          mk-serve = mkServeCheck;
          hrx-compile = self.packages.${system}.hrx;
          pr = prCheck;
        }
      );
    };
}
