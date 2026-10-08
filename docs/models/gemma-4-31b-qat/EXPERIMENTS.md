# Gemma 4 31B QAT experiments

The QAT checkpoint runs on the standard 31B engine; its kernel decisions,
including the Q4_0 paths, are recorded in the
[31B experiments](../gemma-4-31b/EXPERIMENTS.md).

| Experiment | Decision / evidence |
| --- | --- |
| Drafter for the QAT target | Retained: the repository's Q4_0 QAT `gemma4-assistant` (the file `-hf` auto-discovery selects). Its drafter head is already Q4_0, so the Q8_0 → Q4_K head repack does not apply; it runs through the split-K GEMV. |
| QAT vision sidecar | Retained: the repository's own `mmproj-BF16.gguf`; its weights differ from the standard 31B sidecar (SHA-256 `d904b357…` vs `7a4601b1…`). |
| Binary16 prefill (2026-10-05) | Retained, as for the [standard 31B](../gemma-4-31b/EXPERIMENTS.md): Q4_0 projections take the Gemma routed down kernel's layout in a dense mode (two 18-byte blocks staged as nine words per row and stage, weights (q − 8) d as binary16) and the fused gate/up GeGLU epilogue. Cold 2048-row shapes 32.5–33 → 38 TFLOP/s; pp2048 486 → 570 tok/s; prefill KL vs the oracle 0.0011 → 2.2e-6 |
| Sibling drafts (2026-10-05) | Retained with the [31B's](../gemma-4-31b/EXPERIMENTS.md); `gemma4.target` on the QAT pair accepts 22 of 171 siblings over three greedy prompts, every cycle's logits equal to decode's |
