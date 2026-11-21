# Declared models through Gufo

Local: each model as a user runs it, `gufo model install ID`, then `gufo serve llm --model ID --sessions 4`. Strix Halo (gfx1151), Linux 7.0, BIOS performance mode; Gufo ec6d3d07 (Nix) in Docker, stock 8 MiB locked memory (weights preloaded); 2026-11-20.

DevLab: AMD DevLab Dispatch Strix Halo runners, the same user flow: `gufo model install ID` with a Gufo bundle of this extension on Gufo main, then `conformance.py --logits` on the installed model (what `--model ID` resolves to); loom-fun/loom-skills#1335, runs 44841985936 (v0.3.2 on 029a156; 2026-11-20) and 44911502685 (the 4 models added in 0.3.6; 2026-11-21). The first 23 also passed the same checks earlier from loom-fetch packages (runs 44833137889, 44838012335).

Gufo checks: `conformance.py --logits` (29 checks). Throughput: `bench.py --reps 2` (greedy, thinking off;
decode = one stream after a short prompt, prefill = time to first token of a ~10k-token document, 4 streams =
aggregate decode). Gufo / loom-serve: the same package, preset and flags under loom-serve 0.2.0 on the same machine.

| Model ID | Run | Gufo checks | Decode tok/s | Prefill tok/s | 4 streams tok/s | Gufo / loom-serve decode, prefill, 4 streams | Notes |
|---|---|---|---|---|---|---|---|
| `qwen3-0.6b-loom` | local | 29/29 | 344.4 | 8,674 | 1,141.2 |  |  |
| `qwen2.5-0.5b-instruct-loom` | local | 29/29 | 366.9 | 14,099 | 1,292.4 |  |  |
| `qwen3.5-0.8b-loom` | local | 29/29 | 258.5 | 11,731 | 764.2 |  |  |
| `gemma-3-1b-it-loom` | local | 21/28 | 207.7 | 11,190 | 663.9 |  | no tool calls (model) |
| `qwen2.5-1.5b-instruct-loom` | local | 29/29 | 182.5 | 5,894 | 639.5 |  |  |
| `lfm3-omni-2b-loom` | local | 29/29 | 172.3 | 7,329 | 614.5 |  |  |
| `lfm3-1.7b-loom` | local | 29/29 | 172.0 | 7,170 | 608.6 |  |  |
| `hyperclovax-seed-text-instruct-1.5b-loom` | local | 21/28 | 181.1 | 5,969 | 642.3 |  | no tool calls (model) |
| `qwen3-1.7b-loom` | local | 29/29 | 164.7 | 5,153 | 546.0 |  |  |
| `qwen3.5-2b-loom` | local | 28/29 | 145.0 | 6,442 | 383.5 |  | reasoning effort: low and high effort both reason to the length limit (the small model's reasoning loops) |
| `llama-3.2-3b-instruct-loom` | local | 28/29 | 98.5 | 2,901 | 309.2 |  | tool round trip: calls the tool again instead of answering (model behavior; loom-serve does the same) |
| `qwen3-4b-instruct-2507-loom` | local | 29/29 | 77.4 | 2,255 | 250.1 |  |  |
| `spark-x2.5-4b-loom` | local | 29/29 | 72.7 | 2,774 | 254.0 |  |  |
| `spark-x2.6-4b-loom` | local | 29/29 | 72.5 | 2,753 | 218.9 |  |  |
| `qwen3-4b-loom` | local | 29/29 | 104.8 | 2,209 | 225.9 |  |  |
| `qwen3-vl-4b-instruct-loom` | local | 29/29 | 78.3 | 2,241 | 266.1 |  |  |
| `qwen3.5-4b-loom` | local | 29/29 | 90.9 | 2,930 | 192.9 |  |  |
| `gemma-4-e2b-it-loom` | local | 29/29 | 151.8 | 3,834 | 396.6 |  |  |
| `qwen2.5-7b-instruct-loom` | local | 29/29 | 48.2 | 1,879 | 163.4 |  |  |
| `qwen2.5-vl-7b-instruct-loom` | local | 21/28 | 47.8 | 1,864 | 168.8 |  | no tool calls (model) |
| `gemma-4-e4b-it-loom` | local | 29/29 | 85.9 | 2,528 | 219.1 |  |  |
| `qwen3-8b-loom` | local | 29/29 | 55.9 | 1,428 | 147.6 |  |  |
| `qwen3-vl-8b-instruct-loom` | local | 29/29 | 45.4 | 1,457 | 154.5 |  |  |
| `ornith-1.0-9b-loom` | local | 29/29 | 41.0 | 1,739 | 141.4 |  |  |
| `ornith-1.5-9b-loom` | DevLab | 29/29 |  |  |  |  |  |
| `nemotron-3.5-bolt-e4b-loom` | DevLab | 29/29 |  |  |  |  |  |
| `qwen3.5-9b-loom` | DevLab | 29/29 |  |  |  |  |  |
| `mellum2.1-12b-a2.5b-thinking-loom` | DevLab | 29/29 |  |  |  |  |  |
| `qwen3-14b-loom` | DevLab | 29/29 |  |  |  |  |  |
| `gpt-oss-20b-loom` | DevLab | 29/29 |  |  |  |  |  |
| `lfm2.5-24b-a2b-loom` | DevLab | 29/29 |  |  |  |  |  |
| `gemma-4-26b-a4b-it-loom` | DevLab | 29/29 |  |  |  |  |  |
| `qwen3-coder-30b-a3b-instruct-loom` | DevLab | 29/29 |  |  |  |  |  |
| `qwen3.8-27b-loom` | DevLab | 29/29 |  |  |  |  |  |
| `qwen3-30b-a3b-instruct-2507-loom` | DevLab | 29/29 |  |  |  |  |  |
| `gemma-4-31b-it-loom` | DevLab | 29/29 |  |  |  |  |  |
| `qwen2.5-32b-instruct-loom` | DevLab | 29/29 |  |  |  |  |  |
| `laguna-xs-2.1-loom` | DevLab | 28/29 |  |  |  |  | shared prefix across sessions: a prefix shared by two different conversations is not reused; each conversation's own history is |
| `nanbeige4.3-33b-loom` | DevLab | 29/29 |  |  |  |  |  |
| `qwen3.6-27b-loom` | DevLab | 29/29 |  |  |  |  |  |
| `qwen3.6-35b-a3b-loom` | DevLab | 29/29 |  |  |  |  |  |
| `ornith-1.0-35b-loom` | DevLab | 28/29 |  |  |  |  | tool_choice named: reasoning loops when the named tool conflicts with the question (as on Loom 0.1.0, results/logits.md) |
| `minicpm-v-4.7-35b-a3b-loom` | DevLab | 29/29 |  |  |  |  |  |
| `sage-a1.5-loom` | DevLab | 29/29 |  |  |  |  |  |
| `qwen3.5-35b-a3b-loom` | DevLab | 29/29 |  |  |  |  |  |
| `ornith-1.5-35b-a3b-loom` | DevLab | 29/29 |  |  |  |  |  |
| `muse-glimmer-30b-loom` | DevLab | 29/29 |  |  |  |  |  |
| `qwen3.8-27b-ud-q4_k_xl-loom` | DevLab | 29/29 |  |  |  |  |  |
| `gemma-4-26b-a4b-it-qat-loom` | DevLab | 29/29 |  |  |  |  |  |
| `gemma-4-31b-it-qat-loom` | DevLab | 29/29 |  |  |  |  |  |
| `flint-loom` | DevLab | 29/29 |  |  |  |  |  |
