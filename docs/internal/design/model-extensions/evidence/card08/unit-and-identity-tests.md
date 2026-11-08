# Card 08 — selected unit, identity and model tests

Selections come from the live CTest inventory of the candidate build
directories (`ctest --test-dir build/card08-<cfg> -N -R …`), restricted to the
tiers this card moved plus the identity goldens. Asset-gated rows are recorded
as skipped, never as passes.

## CPU configuration (`build/card08-cpu`, `ENGINE_ENABLE_HIP=OFF`)

```
nix develop -c ctest --test-dir build/card08-cpu -R '^(eval_regrade_test|eval_http_test|eval_http_client_test|minimax_h3_runtime_test|minimax_h3_tokenizer_test|minimax_h3_sampling_test|minimax_h3_generation_test|minimax_h3_video_vae_plan|minimax_h3_media_test|ds4\.sampling|ds4\.template|ds4\.cli|ds4\.dataset|ds4\.eval|openai_chat_test|qwen_oracle_test|qwen_l1_norm_module_test|qwen_l1_ffn_module_test|qwen_l1_ssm_module_test|extension_target_ownership_gate|extension_isolation_report|extension_core_model_includes_gate|extension_family_graph_gate|check_extension_isolation_test|check_docs_test)$' --output-on-failure
```

`100% tests passed out of 25`, exit 0, 28.80 s wall. No skips. This is the
first run in which `ds4.dataset` and `ds4.eval` read the core fixture
`tests/fixtures/eval/antirez-ds4.json`, `ds4.sampling` / `ds4.template` /
`openai_chat_test` link `gufo_deepseek_v4_flash_text`, and the three
`qwen_l1_*_module_test` targets plus `qwen_oracle_test` link `gufo_compare`.

The first attempt of this selection failed on `ds4.dataset`:
`FileNotFoundError: …/tests/fixtures/eval/official-0731.json`. The test had
located the official 0731 continuation fixture with
`self.path.with_name(...)`, so moving the eval dataset moved its sibling
lookup. `tests/models/deepseek_v4_flash/dataset_test.py` now keeps an explicit
`cls.model_fixtures = Path(__file__).parent / "fixtures"` for that fixture and
only reads the eval dataset from the core location; the fixture itself did not
move (its schema is `gufo.ds4-official-continuations.v1`, it is referenced by
`tools/ds4/reference.py`, `tests/models/deepseek_v4_flash/CMakeLists.txt` and
`THIRD_PARTY_NOTICES.md`). After the fix the full selection above passes.

## HIP configuration (`build/card08-gpu`)

```
nix develop -c ctest --test-dir build/card08-gpu -R '^(qwen27b_continuation_identity_test|qwen38_flash_next_continuation_identity_test|deepseek_v4_flash_continuation_identity_test|minimax_h3_[a-z0-9_]*|video_(cli|jobs|api)_test|ds4\.(sampling|template|cli|dataset|eval))$' --output-on-failure
```

`100% tests passed out of 30`, exit 0. 21 rows passed, including the three
continuation identity goldens:

- `qwen27b_continuation_identity_test`,
  `qwen38_flash_next_continuation_identity_test`,
  `deepseek_v4_flash_continuation_identity_test` — Passed
- `minimax_h3_dit_analytic`, `minimax_h3_prompt_encoder_analytic`,
  `minimax_h3_sampling_hip_test`, `minimax_h3_audio_vae_primitives`,
  `minimax_h3_generation_test`, `minimax_h3_media_test`,
  `minimax_h3_tokenizer_test`, `minimax_h3_runtime_test`,
  `minimax_h3_sampling_test`, `minimax_h3_video_vae_plan`,
  `video_cli_test`, `video_jobs_test`, `video_api_test`,
  `ds4.sampling`, `ds4.template`, `ds4.cli`, `ds4.dataset`, `ds4.eval` — Passed

Nine H3 rows are **skipped, not passed** (asset gate, exit 77):
`minimax_h3_runtime_hip_test`, `minimax_h3_prompt_encoder_real`,
`minimax_h3_dit_block0_real`, `minimax_h3_denoiser_smoke`,
`minimax_h3_denoiser_oracle`, `minimax_h3_denoiser_production_shape_oracle`,
`minimax_h3_denoiser_reuse_smoke`, `minimax_h3_video_vae_real`,
`minimax_h3_audio_vae_real`. A direct run reports the reason:
`SKIP: set GUFO_H3_MODEL_ROOT and GUFO_H3_SOURCE_MANIFEST`; no MiniMax H3
weights are provisioned under `models/` on this host, so the moved H3 HIP
numerical checks stay in
[pending-commands.md](pending-commands.md).

## Sanitizer, assertion and code generation state (integration correction)

The original assertion/parity claim was wrong. The three support targets
`gufo_minimax_h3`, `gufo_eval`, `gufo_compare` are created before the test-scope
`add_compile_options(-UNDEBUG)`; only `gufo_deepseek_v4_flash_text` explicitly
gets `-UNDEBUG` under `BUILD_TESTING`.

| Configuration | H3/eval/compare | DeepSeek text | Sanitizers |
| --- | --- | --- | --- |
| Production `build/card08-release` (RelWithDebInfo, tests OFF) | `NDEBUG` | `NDEBUG`, shipped `-O3 -ffast-math -fno-finite-math-only` | OFF |
| HIP-test `build/card08-gpu` (RelWithDebInfo) | `NDEBUG` | `-UNDEBUG`, shipped optimization; these assertion-armed objects now reach test-build `gufo`, whereas the parent linked `NDEBUG` model objects | OFF |
| CPU-test `build/card08-cpu` (RelWithDebInfo) | `NDEBUG` | `-UNDEBUG`; after integration, configuration `-O2`, no fast-math | OFF |
| Fresh Debug CPU sanitizer (`/tmp/card08-sanitizer`) | No target `-UNDEBUG`; Debug normally has no `NDEBUG` in the first place | explicit `-UNDEBUG`, Debug optimization, no fast-math | PENDING: configure/build/run blocked before cache creation |

The worker unconditionally applied `-O3 -ffast-math` to CPU-only text objects.
That was a real change from the parent's duplicated test copies (`-O2`, no
fast-math). Keeping production fast-math in CPU-only sampler tests is not
acceptable here: it changes numerically sensitive test code generation without
qualification. Integration scopes those options to `ENGINE_ENABLE_HIP`, while
retaining one implementation per source. The existing CPU cache and artifacts
predate this fix and do not validate it; the fresh build remains PENDING.
HIP-test assertion sharing is intentional: one library serves the executable
and tests, and assertions stay armed in tests. Production flags are unchanged.

Sanitizer provenance is static, not an execution pass: the H3 CPU, eval and
comparison sources used to inherit `-fsanitize=address,undefined` from `gufo_core`
and still inherit the same directory options after extraction. Parent CPU-only
DeepSeek text test copies inherited those sanitizer options; the new text target
also does. Parent HIP production text objects were created before the sanitizer
block; the new shared text objects inherit it when sanitizers are enabled.
No sanitizer-enabled candidate compile database was produced in this session.
Thus per-source actual sanitizer flags/cache/test results remain PENDING;
[integration-reruns.txt](integration-reruns.txt) contains all exact commands and
the daemon denial. An OFF cache is not sanitizer execution evidence.
