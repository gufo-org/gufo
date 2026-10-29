Text (`src/ext/text.hpp`): `TextRunnerOptions::kv_cache_bytes` carries
`gufo serve --kv-cache-bytes`, the device bytes of K/V rows all sessions may
hold at once for runners that grow each session's K/V with its tokens
(Qwen3.8 Flash-Next). Zero (the default) keeps every session's full context;
other runners ignore it. Additive; designated initializers keep the default.
