# Flash-Next switch-over: remaining work

Unfinished acceptance gates for [PR #543](https://github.com/gufo-org/gufo/pull/543), the
Flash-only [card 19](19-switch-over.md) migration. Passing results on older
revisions remain historical evidence until the affected checks are repeated on
the final revision. Restore latency from disk and a Flash server sharing a
`--cache-disk` directory with legacy-cache models are not merge gates.

| Priority | What is missing | Required completion / current limit |
| --- | --- | --- |
| High | Check the other models on the merged head | Build and run the GPU serving tests `inference_backend_gpu_test`, `qwen_vision_serving_test` and `ds4.serving`. Run Qwen 27B `cache-growth`, `cache-edits` and `cache` on the candidate, with matched main controls for affected timings. |
| High | Resolve small-budget retention timing flags | Reuse and cold outputs pass (card 22); two alternations are retained. Fix or explain the reproduced 5% and 3 ms flags: reassignment queue time moving between requests, the synchronous post-token retry capture in fan-out (267 ms AR gap), `depth_unchanged` +7.5 ms prefill, the MTP growth cold-control prefill, and the MTP concurrency probe losing its exact retry copy. Re-measure against the merged main. |
| High | Replay the reported deep/short-peer workflow | Run the approximately 175K-token restart history, three rounds of four short same-system conversations, and advancing deep turns in **AR and MTP**, checking publication, over-RAM restore correctness, actual drafts and all cold controls. Restore latency from disk is not a gate. The harness exists; production qualification remains. |
| High | Finish isolated crash/restart and lower-budget investigation | Repeat the deep SIGKILL recovery test without competing builds so every cold comparison finishes. Investigate the separate lower-effective-RAM cold-control timeout; neither is a complete pass. |
| High | Resolve affected serving latency flags | Investigate queueing, maximum token gap, MTP concurrency TTFT and same-slot restore/reassignment. The reassignment queue cost is one synchronous 256 KiB chunk copy per round trip (about 9 ms residual wait); fixing it needs an in-place same-slot restore or a measured reassignment harness. Retain every result and use matched main controls for affected requests. Correctness passing alone does not qualify performance. |
| High | Validate the final revision across remaining workflows | On the merged head, run relevant `cache`, `long-context`, state-edge/cancellation, Messages API and transformation checks, plus the real Pi 0.87 tool/edit/bugfix workflow. Include the `cache-bridge` → `cache-pressure` history at 2 GiB and the six-suite history at 8 GiB. Verify loaded mode and actual MTP drafts; inspect actual tool results. Add or extend integration tests for fixes that need regression protection. |
| Medium | Complete issue reproduction coverage | The **160-issue inventory is complete** and retained in `/home/mixer/gufo-qualification/pr543-review-2026-10-11`; its pressure rows (#331, #335, #451, #462) predate `f9528f3c`. Partially verified and unverified cache reproductions still need evidence. Update each assessment with its revision, result and limits. Keep unsupported features and issues outside Flash scope explicit. |
| Medium | Finish resource and benchmark gates | Qualify slot reassignment/spill behavior, idle/foreground timing and the standard pp2048/tg128 benchmark against the merged main (#551 changes Flash drafting). Demonstrate any claimed RAM reduction. Re-confirm card 14's AR/MTP capacity rows, measured with 4 GiB of committed backing, at the startup-committed automatic budget (D4). Startup-committed backing leaves less RAM for co-resident model, image, ASR and TTS servers; their KFD capacity remains unverified. Earlier benchmark evidence is inconclusive. |
| Final | Consolidate final evidence and checks | After each conclusive fix, push it and update the pull request description. On the final revision, complete applicable contracts/formatting/CI, update the cards, `todo.md` and the issue audit, and publish an explicit pass/fail/inconclusive acceptance record. |

Qwen 27B and DeepSeek migration, legacy cache deletion and the full RFC
all-model cutover remain outside this Flash-only change.
