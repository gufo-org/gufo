{
  description = "gufo - Gufo Engine for AMD Strix Halo (gfx1151 GPU)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # Reference ASR/TTS runtime for benchmark comparisons; exposes a
    # ROCm gfx1151 package tuned for Strix Halo.
    audio-cpp = {
      url = "github:0xShug0/audio.cpp";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self, nixpkgs, audio-cpp }:
    let
      # Strix Halo is a Linux x86-64-only target (see README non-goals).
      forAllSystems = nixpkgs.lib.genAttrs [ "x86_64-linux" ];
      version = self.shortRev or self.dirtyShortRev or "dirty";

      pkgs = forAllSystems (system: import nixpkgs { inherit system; });

      gufoPackages = forAllSystems (
        system:
        pkgs.${system}.callPackage ./.devops/nix/scope.nix {
          inherit version;
          pkgs = pkgs.${system};
        }
      );

      llamaReferences = forAllSystems (
        system: pkgs.${system}.callPackage ./.devops/nix/llama-cpp-reference.nix { }
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

      # Model reference and client-validation tools; never a transitive
      # dependency of the server (see tools/README.md for the offline toolchain).
      # Unified python313 + torchWithRocm: every Strix Halo box ships ROCm, so
      # the single toolchain serves CPU flows and the --device cuda
      # reference forward alike. gfx1151 verified on this host.
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
            ps.openai # official SDK for local API compatibility checks
            lpipsRocm
            ps.numpy
            ps.scipy
            ps.matplotlib # benchmark charts (tools/bench/model-bench.py render)
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
          default = base;
          tp2-rdma = gufoPackages.${system}.tp2-rdma;
          # Optional benchmark packages: excluded from Gufo, the default
          # development shell and hosted checks.
          ds4-reference = pkgs.${system}.callPackage ./.devops/nix/ds4-reference.nix { };
          llama-cpp-reference = llamaReferences.${system}.release;
          llama-cpp-mtp-reference = llamaReferences.${system}.mtp;
        }
      );

      devShells = forAllSystems (
        system:
        {
          # Unified toolchain (CPU + ROCm torch): one default shell serves all
          # offline CPU flows and the ROCm reference forward.
          default = pkgs.${system}.mkShell {
            inputsFrom = [ self.packages.${system}.default ];
            packages = [
              (pythonTools system)
              pkgs.${system}.clang-tools
              pkgs.${system}.ffmpeg-headless
              pkgs.${system}.sox
              pkgs.${system}.rocmPackages.rocprofiler-sdk
              pkgs.${system}.sqlite
              audio-cpp.packages.${system}.rocm-gfx1151
            ];
            env = {
              TORCH_HOME = "${alexnetTorchHome system}";
              LD_LIBRARY_PATH = pkgs.${system}.lib.makeLibraryPath [
                pkgs.${system}.stdenv.cc.cc.lib
                pkgs.${system}.zlib
              ];
            };
          };
          tp2-rdma = pkgs.${system}.mkShell {
            inputsFrom = [ self.packages.${system}.tp2-rdma ];
            packages = [ pkgs.${system}.clang-tools ];
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
            files = [
              ".clang-format"
              "tools/ci/check-format.py"
            ];
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
              "tools"
            ];
            files = [
              ".clang-format"
              "CMakeLists.txt"
              "tools/prof/prof.py"
              "tools/bench/speculative-corpus.py"
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
            nativeBuildInputs = [ pkgsSys.clang-tools pkgsSys.python3 ];
            src = formatSource;
          } ''
            cd "$src"
            python3 tools/ci/check-format.py
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
              pkgsSys.curl
              pkgsSys.libpng
              pkgsSys.libjpeg
              pkgsSys.openssl
            ];
            src = staticAnalysisSource;
          } ''
            export HOME=$TMPDIR
            mkdir -p build && cd build
            cmake "$src" -GNinja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTING=OFF -DENGINE_ENABLE_HIP=OFF

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

          mkTestCheck = full: pkgsSys.runCommand (if full then "check-tests" else "check-pr-tests") {
            nativeBuildInputs = [
              pkgsSys.stdenv.cc
              pkgsSys.ccache
              pkgsSys.cmake
              pkgsSys.ninja
              (if full then pkgsSys.python3.withPackages (ps: [ ps.numpy ]) else pkgsSys.python3)
              pkgsSys.icu
              pkgsSys.curl
              pkgsSys.libpng
              pkgsSys.libjpeg
              pkgsSys.openssl
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
            cmake "$src" -GNinja $ccache_launcher -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON -DENGINE_ENABLE_HIP=OFF
            ${if full then ''
              cmake --build . --parallel "$NIX_BUILD_CORES"
              ctest --output-on-failure --timeout 120
            '' else ''
              cmake --build . --parallel "$NIX_BUILD_CORES" --target check-pr
            ''}

            if [[ -n "$ccache_launcher" ]]; then
              ccache --show-stats
            fi

            mkdir -p $out
            echo "PASS: ${if full then "Full CPU" else "Hosted CPU contract"} suite passed" > $out/result.txt
          '';

          testCheck = mkTestCheck true;
          prTestCheck = mkTestCheck false;

          mkServeCheck =
            let
              # Test command rendering without realizing the GPU package.
              mkServe = args: self.lib.${system}.mkGufoServe (args // { gufo = "/gufo-test"; });
              ttsCmd = mkServe {
                modality = "tts";
                model = "/var/models/qwen3-tts";
                context = 4096;
                port = 9100;
                servedModelName = "voice-test";
              };
              voicesCmd = mkServe {
                modality = "tts";
                model = "/var/models/qwen3-tts";
                voices = {
                  plain = "/var/voices/plain.wav";
                  inline = {
                    wav = "/var/voices/inline.wav";
                    text = "spoken reference line";
                  };
                  sidecarfile = {
                    wav = "/var/voices/file.wav";
                    text = "/var/voices/file.txt";
                  };
                };
              };
              asrCmd = mkServe {
                modality = "asr";
                model = "/var/models/qwen3-asr";
                context = 1024;
              };
              cmd = mkServe {
                model = "/var/models/qwen.gguf";
                context = 4096;
                servedModelName = "qwen-test";
                speculative = "dflash2";
                dflashModel = "/var/models/qwen-draft.gguf";
                port = 9000;
                temperature = 0.8;
                topK = 40;
                draftPolicy = "fixed";
                topP = 0.9;
                minP = 0.05;
                minKeep = 3;
                seed = 123;
                repeatPenalty = 1.1;
                repeatLastN = 32;
                frequencyPenalty = 0.25;
                presencePenalty = 0.5;
                reasoningEffort = "high";
                preserveThinking = "auto";
                cacheDisk = "/var/cache/gufo";
                cacheDiskBytes = 1024;
                cacheDiskStagingBytes = 512;
              };
              dsparkCmd = mkServe {
                model = "/var/models/ds4.gguf";
                speculative = "dspark";
                dsparkModel = "/var/models/dspark.gguf";
                draftTokens = 3;
              };
              mtpCmd = mkServe {
                model = "/var/models/qwen-flash.gguf";
                speculative = "mtp";
                mtpModel = "/var/models/mtp.gguf";
                draftTokens = 3;
              };
              imageCmd = mkServe {
                modality = "image";
                model = "/var/models/qwen-image";
                servedModelName = "qwen-image-test";
                port = 9200;
                maxRequestBytes = 33554432;
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
              echo "$cmd_str" | grep -F -- "--draft-policy fixed"
              echo "$cmd_str" | grep -F -- "--port 9000"
              echo "$cmd_str" | grep -F -- "--temperature 0.800000"
              echo "$cmd_str" | grep -F -- "--top-k 40"
              echo "$cmd_str" | grep -F -- "--top-p 0.900000"
              echo "$cmd_str" | grep -F -- "--min-p 0.050000"
              echo "$cmd_str" | grep -F -- "--min-keep 3"
              echo "$cmd_str" | grep -F -- "--seed 123"
              echo "$cmd_str" | grep -F -- "--repeat-penalty 1.100000"
              echo "$cmd_str" | grep -F -- "--repeat-last-n 32"
              echo "$cmd_str" | grep -F -- "--frequency-penalty 0.250000"
              echo "$cmd_str" | grep -F -- "--presence-penalty 0.500000"
              echo "$cmd_str" | grep -F -- "--reasoning-effort high"
              echo "$cmd_str" | grep -F -- "--preserve-thinking auto"
              echo "$cmd_str" | grep -F -- "--cache-disk /var/cache/gufo"
              echo "$cmd_str" | grep -F -- "--cache-disk-bytes 1024"
              echo "$cmd_str" | grep -F -- "--cache-disk-staging-bytes 512"

              tts_str="${ttsCmd}"
              echo "$tts_str" | grep -F -- "--port 9100 tts"
              echo "$tts_str" | grep -F -- "tts --model /var/models/qwen3-tts --served-model-name voice-test --context 4096"

              # Voices take a bare WAV (sidecar transcript) or {wav, text},
              # where text is either the transcript or a path holding it.
              voices_str="${voicesCmd}"
              echo "$voices_str" | grep -F -- "--voice 'plain=/var/voices/plain.wav'"
              echo "$voices_str" | grep -F -- "--voice-text 'inline=spoken reference line'"
              echo "$voices_str" | grep -F -- "--voice-text 'sidecarfile=/var/voices/file.txt'"
              test "$(echo "$voices_str" | grep -o -- '--voice ' | wc -l)" = 3

              asr_str="${asrCmd}"
              echo "$asr_str" | grep -F -- "asr --model /var/models/qwen3-asr --context 1024"

              dspark_cmd="${dsparkCmd}"
              echo "$dspark_cmd" | grep -F -- "--speculative dspark"
              echo "$dspark_cmd" | grep -F -- "--dspark-model /var/models/dspark.gguf"
              echo "$dspark_cmd" | grep -F -- "--draft-tokens 3"
              if echo "$dspark_cmd" | grep -E -- '--mtp-model|--draft-policy'; then
                exit 1
              fi
              mtp_cmd="${mtpCmd}"
              echo "$mtp_cmd" | grep -F -- "--speculative mtp"
              echo "$mtp_cmd" | grep -F -- "--mtp-model /var/models/mtp.gguf"
              echo "$mtp_cmd" | grep -F -- "--draft-tokens 3"
              image_str="${imageCmd}"
              echo "$image_str" | grep -F -- "--port 9200"
              echo "$image_str" | grep -F -- "--max-request-bytes 33554432"
              echo "$image_str" | grep -F -- "image --model /var/models/qwen-image --served-model-name qwen-image-test"
              mkdir -p $out
              echo "PASS: mkGufoServe CLI string check passed" > $out/result.txt
            '';

          # The hosted runner builds CPU contracts only. GPU compilation,
          # whole-tree clang-tidy, full CPU and H3 oracles are explicit checks.
          prCheck = pkgsSys.runCommand "check-pr" { } ''
            mkdir -p $out
            cat ${formatCheck}/result.txt \
                ${dependencyInventoryCheck}/result.txt \
                ${docsCheck}/result.txt \
                ${prTestCheck}/result.txt \
                ${mkServeCheck}/result.txt > $out/pr-summary.txt
            cat $out/pr-summary.txt
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
          pr-tests = prTestCheck;
          mk-serve = mkServeCheck;
          pr = prCheck;
        }
      );
    };
}
