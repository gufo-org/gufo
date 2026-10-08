
#### Flash-Next (MTP)

| Operation | Depth | Today | Hybrid |
| --- | ---: | ---: | ---: |
| Checkpoint bytes | 32k | 1.00 GB | 119 MB + shared KV |
| Capture (on the request path) | 32k | 3 ms | ~3 ms |
| Persist one checkpoint | 32k | 2.3 s | 399 ms |
| Restore from RAM | 32k | 10 ms | 10 ms |
| Restore from disk (cold) | 32k | 907 ms | 1.0 s |
| Prefill instead (cold) | 32k | 25.8 s | 25.8 s |
| Checkpoints of this conversation in the RAM budget | 32k | 9 | 56 |
| Checkpoint bytes | 100k | 2.87 GB | 119 MB + shared KV |
| Capture (on the request path) | 100k | 3 ms | ~3 ms |
| Persist one checkpoint | 100k | 6.5 s (skipped: > staging) | 399 ms |
| Restore from RAM | 100k | 29 ms | 29 ms |
| Restore from disk (cold) | 100k | 2.6 s | 2.9 s |
| Prefill instead (cold) | 100k | 80.4 s | 80.4 s |
| Checkpoints of this conversation in the RAM budget | 100k | 3 | 43 |
| Checkpoint bytes | 149k | 4.21 GB | 119 MB + shared KV |
| Capture (on the request path) | 149k | 3 ms | ~3 ms |
| Persist one checkpoint | 149k | 9.6 s (skipped: > staging) | 399 ms |
| Restore from RAM | 149k | 42 ms | 42 ms |
| Restore from disk (cold) | 149k | 3.8 s | 4.3 s |
| Prefill instead (cold) | 149k | 2.0 min | 2.0 min |
| Checkpoints of this conversation in the RAM budget | 149k | 2 | 34 |

#### 27B (DFlash2)

| Operation | Depth | Today | Hybrid |
| --- | ---: | ---: | ---: |
| Checkpoint bytes | 32k | 2.34 GB | 244 MB + shared KV |
| Capture (on the request path) | 32k | 106 ms | ~14 ms |
| Persist one checkpoint | 32k | 5.3 s | 859 ms |
| Restore from RAM | 32k | 23 ms | 23 ms |
| Restore from disk (cold) | 32k | 2.1 s | 2.4 s |
| Prefill instead (cold) | 32k | 94.0 s | 94.0 s |
| Checkpoints of this conversation in the RAM budget | 32k | 9 | 67 |
| Checkpoint bytes | 100k | 6.80 GB | 244 MB + shared KV |
| Capture (on the request path) | 100k | 181 ms | ~14 ms |
| Persist one checkpoint | 100k | 15.4 s (skipped: > staging) | 859 ms |
| Restore from RAM | 100k | 68 ms | 68 ms |
| Restore from disk (cold) | 100k | 6.2 s | 7.0 s |
| Prefill instead (cold) | 100k | 5.8 min | 5.8 min |
| Checkpoints of this conversation in the RAM budget | 100k | 3 | 52 |
| Checkpoint bytes | 149k | 10.01 GB | 244 MB + shared KV |
| Capture (on the request path) | 149k | 236 ms | ~14 ms |
| Persist one checkpoint | 149k | 22.7 s (skipped: > staging) | 859 ms |
| Restore from RAM | 149k | 100 ms | 100 ms |
| Restore from disk (cold) | 149k | 9.1 s | 10.3 s |
| Prefill instead (cold) | 149k | 9.7 min | 9.7 min |
| Checkpoints of this conversation in the RAM budget | 149k | 2 | 42 |
