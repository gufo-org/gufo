# loom-fun packages through Gufo

Local: each model as a user runs it, `gufo model install`, then `gufo serve llm --model <package> --sessions 4`. Strix Halo (gfx1151), Linux 7.0, BIOS performance mode; Gufo ec6d3d07 (Nix) in Docker, stock 8 MiB locked memory (weights preloaded); 2026-11-20.

DevLab: AMD DevLab Dispatch Strix Halo runners, the same user flow: `gufo model install` with a Gufo bundle of this extension on Gufo main, then `conformance.py --logits` on the installed model (what `--model` resolves to); loom-fun/loom-skills#1335, runs 44841985936 (v0.3.2 on 029a156; 2026-11-20) and 44911502685 (the 4 models added in 0.3.6; 2026-11-21). The first 23 also passed the same checks earlier from loom-fetch packages (runs 44833137889, 44838012335).

Gufo checks: `conformance.py --logits` (29 checks). Throughput: `bench.py --reps 2` (greedy, thinking off;
decode = one stream after a short prompt, prefill = time to first token of a ~10k-token document, 4 streams =
aggregate decode). Gufo / loom-serve: the same package, preset and flags under loom-serve 0.2.0 on the same machine.

| Package | Run | Gufo checks | Decode tok/s | Prefill tok/s | 4 streams tok/s | Gufo / loom-serve decode, prefill, 4 streams | Notes |
|---|---|---|---|---|---|---|---|
| `loom-fun/unsloth-Qwen3-0.6B-loom` | local | 29/29 | 344.4 | 8,674 | 1,141.2 |  |  |
| `loom-fun/bartowski-Qwen2.5-0.5B-Instruct-loom` | local | 29/29 | 366.9 | 14,099 | 1,292.4 |  |  |
| `loom-fun/unsloth-Qwen3.5-0.8B-loom` | local | 29/29 | 258.5 | 11,731 | 764.2 |  |  |
| `loom-fun/ggml-org-gemma-3-1b-it-loom` | local | 21/28 | 207.7 | 11,190 | 663.9 |  | no tool calls (model) |
| `loom-fun/bartowski-Qwen2.5-1.5B-Instruct-loom` | local | 29/29 | 182.5 | 5,894 | 639.5 |  |  |
| `loom-fun/ggml-org-LFM3-Omni-2B-loom` | local | 29/29 | 172.3 | 7,329 | 614.5 |  |  |
| `loom-fun/bartowski-LFM3-1.7B-loom` | local | 29/29 | 172.0 | 7,170 | 608.6 |  |  |
| `loom-fun/rippertnt-HyperCLOVAX-SEED-Text-Instruct-1.5B-loom` | local | 21/28 | 181.1 | 5,969 | 642.3 |  | no tool calls (model) |
| `loom-fun/ggml-org-Qwen3-1.7B-loom` | local | 29/29 | 164.7 | 5,153 | 546.0 |  |  |
| `loom-fun/unsloth-Qwen3.5-2B-loom` | local | 28/29 | 145.0 | 6,442 | 383.5 |  | reasoning effort: low and high effort both reason to the length limit (the small model's reasoning loops) |
| `loom-fun/unsloth-Llama-3.2-3B-Instruct-loom` | local | 28/29 | 98.5 | 2,901 | 309.2 |  | tool round trip: calls the tool again instead of answering (model behavior; loom-serve does the same) |
| `loom-fun/unsloth-Qwen3-4B-Instruct-2507-loom` | local | 29/29 | 77.4 | 2,255 | 250.1 |  |  |
| `loom-fun/XHToken-Spark-X2.5-4B-loom` | local | 29/29 | 72.7 | 2,774 | 254.0 |  |  |
| `loom-fun/bartowski-Spark-X2.6-4B-loom` | local | 29/29 | 72.5 | 2,753 | 218.9 |  |  |
| `loom-fun/ggml-org-Qwen3-4B-loom` | local | 29/29 | 104.8 | 2,209 | 225.9 |  |  |
| `loom-fun/unsloth-Qwen3-VL-4B-Instruct-loom` | local | 29/29 | 78.3 | 2,241 | 266.1 |  |  |
| `loom-fun/unsloth-Qwen3.5-4B-loom` | local | 29/29 | 90.9 | 2,930 | 192.9 |  |  |
| `loom-fun/unsloth-gemma-4-E2B-it-loom` | local | 29/29 | 151.8 | 3,834 | 396.6 |  |  |
| `loom-fun/bartowski-Qwen2.5-7B-Instruct-loom` | local | 29/29 | 48.2 | 1,879 | 163.4 |  |  |
| `loom-fun/ggml-org-Qwen2.5-VL-7B-Instruct-loom` | local | 21/28 | 47.8 | 1,864 | 168.8 |  | no tool calls (model) |
| `loom-fun/unsloth-gemma-4-E4B-it-loom` | local | 29/29 | 85.9 | 2,528 | 219.1 |  |  |
| `loom-fun/unsloth-Qwen3-8B-loom` | local | 29/29 | 55.9 | 1,428 | 147.6 |  |  |
| `loom-fun/unsloth-Qwen3-VL-8B-Instruct-loom` | local | 29/29 | 45.4 | 1,457 | 154.5 |  |  |
| `loom-fun/unsloth-Ornith-1.0-9B-loom` | local | 29/29 | 41.0 | 1,739 | 141.4 |  |  |
| `loom-fun/bartowski-Ornith-1.5-9B-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/unsloth-Qwen3.5-9B-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/bartowski-Mellum2.1-12B-A2.5B-Thinking-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/ggml-org-Qwen3-14B-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/unsloth-gpt-oss-20b-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/ggml-org-LFM2.5-24B-A2B-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/lmstudio-community-gemma-4-26B-A4B-it-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/ggml-org-Qwen3-Coder-30B-A3B-Instruct-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/bartowski-Qwen3.8-27B-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/unsloth-Qwen3-30B-A3B-Instruct-2507-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/unsloth-gemma-4-31B-it-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/bartowski-Qwen2.5-32B-Instruct-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/ggml-org-Laguna-XS-2.1-loom` | DevLab | 28/29 |  |  |  |  | shared prefix across sessions: a prefix shared by two different conversations is not reused; each conversation's own history is |
| `loom-fun/bartowski-Nanbeige4.3-33B-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/ggml-org-Qwen3.6-27B-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/ggml-org-Qwen3.6-35B-A3B-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/lmstudio-community-Ornith-1.0-35B-loom` | DevLab | 28/29 |  |  |  |  | tool_choice named: reasoning loops when the named tool conflicts with the question (as on Loom 0.1.0, results/logits.md) |
| `loom-fun/bartowski-MiniCPM-V-4.7-35B-A3B-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/InternScience-Sage-A1.5-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/unsloth-Qwen3.5-35B-A3B-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/bartowski-Ornith-1.5-35B-A3B-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/ggml-org-Muse-Glimmer-30B-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/unsloth-Qwen3.8-27B-UD-Q4_K_XL` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/unsloth-gemma-4-26B-A4B-it-qat-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/unsloth-gemma-4-31B-it-qat-loom` | DevLab | 29/29 |  |  |  |  |  |
| `loom-fun/ggml-org-Flint-loom` | DevLab | 29/29 |  |  |  |  |  |
