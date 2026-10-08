| Run | Measured prefill | Today | Phase 0 | Hybrid | Hybrid + dense | Reuse match (within 64 tokens) | Refusals: server / simulated today |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash-Next W1, restart before request 12 | — | 137 s | 137 s | 137 s | 137 s | — | — |
| Flash-Next W1, restart before request 20 | — | 161 s | 137 s | 137 s | 137 s | — | — |
| Flash-Next W1, restart before request 28 | — | 185 s | 137 s | 137 s | 137 s | — | — |
| Flash-Next W1 | 130 s | 137 s | 137 s | 137 s | 137 s | 36 | 39 / 32 |
| Flash-Next W2 | 85 s | 87 s | 71 s | 71 s | 70 s | 29 | 22 / 4 |
| Flash-Next W3 | 108 s | 106 s | 106 s | 106 s | 105 s | 47 | 41 / 19 |
| Flash-Next W4 | 108 s | 100 s | 100 s | 100 s | 95 s | 31 | 27 / 14 |
| Flash-Next W2-C4 | 72 s | 71 s | 71 s | 71 s | 70 s | 34 | 27 / 1 |
| Flash-Next W3-C4 | 159 s | 155 s | 155 s | 155 s | 154 s | 88 | 77 / 25 |
| 27B W1, restart before request 12 | — | 597 s | 597 s | 597 s | 597 s | — | — |
| 27B W1, restart before request 20 | — | 718 s | 597 s | 597 s | 597 s | — | — |
| 27B W1, restart before request 28 | — | 855 s | 597 s | 597 s | 597 s | — | — |
| 27B W1 | 589 s | 596 s | 596 s | 596 s | 596 s | 36 | 39 / 32 |
| 27B W2 | 293 s | 288 s | 231 s | 231 s | 229 s | 19 | 19 / 0 |
| 27B W3 | 352 s | 339 s | 339 s | 339 s | 336 s | 41 | 42 / 20 |
| 27B W4 | 403 s | 399 s | 399 s | 363 s | 345 s | 31 | 31 / 9 |
| 27B W2-C4 | 230 s | 228 s | 228 s | 228 s | 226 s | 30 | 22 / 1 |
| 27B W3-C4 | 478 s | 565 s | 483 s | 480 s | 478 s | 67 | 92 / 36 |

| Run | Disk written: server | Today (sim) | Phase 0 | Hybrid | Hybrid + dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash-Next W1 | 7.06 GB | 7.0 GB | 24.1 GB | 5.7 GB | 5.7 GB |
| Flash-Next W2 | 16.33 GB | 15.1 GB | 14.4 GB | 5.0 GB | 5.5 GB |
| Flash-Next W3 | 13.73 GB | 13.5 GB | 13.5 GB | 6.3 GB | 6.2 GB |
| Flash-Next W4 | 23.22 GB | 26.2 GB | 28.5 GB | 6.7 GB | 6.6 GB |
| Flash-Next W2-C4 | 14.48 GB | 14.2 GB | 14.2 GB | 5.1 GB | 5.7 GB |
| Flash-Next W3-C4 | 16.17 GB | 16.1 GB | 16.1 GB | 8.9 GB | 8.7 GB |
| 27B W1 | 16.53 GB | 16.5 GB | 56.8 GB | 13.2 GB | 13.2 GB |
| 27B W2 | 37.93 GB | 35.1 GB | 33.3 GB | 11.2 GB | 12.5 GB |
| 27B W3 | 31.68 GB | 31.5 GB | 31.5 GB | 13.5 GB | 13.1 GB |
| 27B W4 | 70.93 GB | 65.4 GB | 65.4 GB | 15.2 GB | 14.9 GB |
| 27B W2-C4 | 36.13 GB | 33.3 GB | 33.3 GB | 10.9 GB | 12.2 GB |
| 27B W3-C4 | 33.7 GB | 36.2 GB | 35.0 GB | 20.1 GB | 19.7 GB |
