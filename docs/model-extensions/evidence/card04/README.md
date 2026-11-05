# Card 04 implementation evidence

Implementation agent `/root/card04_implement`, parent `d6d0fd4a`. Independent
review by `/root/card04_review` and integration by `/root/card04_integrate`
are complete. Source/representation decisions
are in [INFRASTRUCTURE.md](../../INFRASTRUCTURE.md).

Local Linux x86-64, pinned `nix develop`, GCC 16.2.0. CPU configuration uses
`RelWithDebInfo`, `BUILD_TESTING=ON`, `ENGINE_ENABLE_HIP=OFF`. The existing CPU
cache initially used GCC 15.3 and picked up newer codec libraries on reconfigure;
its first HTTP link failed on `GLIBC_2.43` references. No code workaround was
made. `nix develop -c cmake --fresh --preset cpu-test` refreshed the entire cache
with the pinned GCC 16.2 toolchain; all commands below then passed.

- `nix develop -c cmake --build --preset cpu-test --target
  extension_runtime_test extension_role_test trace_test openai_chat_test
  http_server_test --parallel 4`: passed, 145 compilation/link steps after the
  fresh configure. Later focused extension rebuilds after generator/test changes
  passed. Existing missing-field-initializer warnings in unrelated serving/model
  code remain; the new runtime/generated code has none.
- `nix develop -c ctest --test-dir build/cpu-test -R
  '^(extension_(runtime|role|metadata|schema)_test|trace_test|openai_chat_test|http_server_test)$'
  --output-on-failure`: implementation passed 7/7, zero skips; final integration
  expanded the expression to `^(extension_.*|trace_test|openai_chat_test|http_server_test)$`
  and passed 8/8, including the isolation report. Retained output is
  [focused-ctest.txt](focused-ctest.txt). HTTP tests cover the shared logger's
  threshold, escaping and formatted/request logging, and trace tests cover the
  moved trace sink. The runtime fixture also checks the injected/default Host
  logger and shares state with the old forwarding include path.
- `nix develop -c python3 tests/extensions/metadata_test.py --diagnostics
  docs/model-extensions/evidence/card04/configure-diagnostics.json`: 12 groups
  passed after integration, retaining [113 exact configure diagnostics and three
  edge-fixture configure/build/CTest outcomes](configure-diagnostics.json).
  Cases cover missing required fields, names/IDs/entry-symbol/generated-identifier
  collisions, missing/cyclic and platform-incompatible dependencies, incompatible
  API minor/revision floors, impossible/unknown conditions, missing targets,
  target-definition contracts, shared option types, every core-ownership category,
  library restrictions and deterministic generated output. Three additional
  fixture configurations compare generated files and check excluded versus
  available registration source; they do not require Python discovery or HIP.
- `nix develop -c python3 tests/extensions/metadata_test.py --schema`: full
  Draft 2020-12 schema check passed, including every broader RFC optional
  packaging/docs/CI metadata field and independent malformed-field cases.
  The same schema test is part of the bounded hosted CPU check; its Nix Python
  environment now contains `jsonschema`.
- `nix develop -c cmake -S tests/extensions/fixture -B
  build/extensions-card04-available -DENGINE_ENABLE_HIP=ON
  -DCMAKE_BUILD_TYPE=Release`, then `nix develop -c cmake --build
  build/extensions-card04-available --target extension_runtime_test
  extension_role_test --parallel 4`, then `nix develop -c ctest --test-dir
  build/extensions-card04-available -R '^extension_(runtime|role)_test$'
  --output-on-failure`: passed, 2/2 zero skips. After adding the inert-metadata
  string case, the runtime target rebuilt and executed successfully. This is a
  synthetic C++ provider with its HIP build condition enabled, not GPU execution.
- `nm -C` probes under `nix develop` checked actual linked executables, with
  [binary hashes and results](symbol-probe.json). The CPU executable contains
  neither `CreateUnavailableHipProvider()` nor its provider registration symbol;
  the available variant contains both. The tests use ordinary static archives
  without whole-archive or static registration constructors.
- `nix develop -c cmake -S . -B build/extensions-card04-platform -G Ninja
  -DGUFO_PLATFORM_ONLY=ON -DBUILD_TESTING=OFF`, then `nix develop -c cmake
  --build build/extensions-card04-platform --target gufo_platform gufo_cache
  --parallel 4`: passed, 22 steps. This is Linux platform-only preservation,
  not Windows execution evidence.
- `nix develop -c cmake --preset release`, then `nix develop -c cmake
  --build --preset release --target gufo_logging gufo_ext_runtime --parallel 4`:
  passed, six steps, using the existing pinned gfx1151/HIP configuration,
  GCC 16.2 and HIP Clang 24/ROCm 10.1, `RelWithDebInfo`, `BUILD_TESTING=OFF`.
  No model kernels or inference ran; this proves these infrastructure libraries
  also build in the production configuration.
- `compile_commands.json` inspection confirms `extension_test_base` has no
  inherited `-UNDEBUG`, while `extension_role_test` has `-UNDEBUG`, including in
  optimized builds. Its source fails compilation if `NDEBUG` remains defined.
- Required `nix shell --inputs-from . nixpkgs#clang-tools -c python3
  tools/ci/check-format.py --fix`, then the same command without `--fix`:
  passed, 723 files. The formatter now includes public `include/` headers.
  `git diff --check` passed.

No model/runtime scheduling/numerical implementation changed, so no broad model
or GPU sweep ran. No performance parity, Windows build, release lock-artifact
qualification or model migration is claimed. Production Nix admission of
`include/` moved forward from card 05 because the always-built runtime requires
it immediately; lock/discovery/release policy and schema production admission
remain card 05. User-owned `RFC.md` was not edited or committed (SHA-256
`2ce2becbc0bf0f844702b624de6827c2bae6eba921ab8a1ea883659fa2f96448`).

Independent review integration resolved all three P2 findings:

- Owner properties now use `get_property(... SET)` and metadata emptiness uses
  explicit comparisons, including provider selection during constant generation.
  The production generator compiles and runs extensions named `off`, `no`,
  `false`, `ignore`, `notfound` and `namespace`, targets/providers of the same
  names, and legal false-like global entries. An option scoped only to the
  false-like provider does not leak to its sibling. Configure negatives retain
  duplicate target/provider detection when the first owning extension has a
  false-like name.
- Generated namespaces are `gufo::ext::generated::extension_<name>`, an injective
  prefix of validated extension names. Global extension/provider entry symbols
  reject C++ keywords in both CMake and schema; tests cover both fields. Existing
  fixture includes/definitions use the documented generated shape.
- An escape-aware scan rejects U+0000 before any `string(JSON)` call. Tests use
  decoded NUL in identities, artifact keys, help, optional docs and object keys,
  plus accepted literal backslash-u0000 help with one/two/three backslashes.
  Full-schema recursion rejects NUL in every string and property name. The
  compiled edge fixture checks literal help survives into actual provider data.

After the namespace update and required formatting, both fixture configurations
were rebuilt with `nix develop -c cmake --build <build-directory> --target
extension_runtime_test extension_role_test --parallel 4`, then executed with
`nix develop -c ctest --test-dir <build-directory> -R
'^extension_(runtime|role)_test$' --output-on-failure`: CPU 2/2 and synthetic
available 2/2 passed. Final `nm -C` probes were rerun against those actual
executables and their new SHA-256 hashes retained. Required formatter recheck
passed 723 files. No logging implementation changed during integration. The
compiled metadata test uses CTest and `--config Release` for multi-configuration
build portability; actual Windows and actual CMake 3.21 execution remain
unproven. Current pinned Linux CMake execution passed.

Documentation validation: `python3 tools/ci/check-docs.py` reports 164 existing
broken links in the unrelated local `hrx-system/` tree. The same validator's
`check_links_and_anchors` was applied explicitly to Card04's `CARDS.md`,
`INFRASTRUCTURE.md`, evidence README and public `CHANGES.md`: four documents,
nine local links/anchors, no errors or malformed fenced JSON. `git diff --check`
passed. The unrelated documentation tree was left untouched.
