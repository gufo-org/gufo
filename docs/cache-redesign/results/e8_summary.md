| Run | Measured prefill | Today | Phase 0 | Hybrid | Hybrid + dense | Reuse, simulated today vs actual | Requests within 64 tokens | Refusals: server / simulated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash-Next W1 | 130 s | 137 s | 137 s | 137 s | 137 s | +0.0% | 36/36 | 39 / 64 |
| Flash-Next W2 | 85 s | 88 s | 72 s | 72 s | 71 s | -0.1% | 35/36 | 22 / 26 |
| Flash-Next W3 | 108 s | 108 s | 107 s | 107 s | 107 s | -0.1% | 58/60 | 41 / 47 |
| Flash-Next W4 | 108 s | 100 s | 100 s | 100 s | 95 s | +0.4% | 28/33 | 27 / 26 |
| Flash-Next W2-C4 | 72 s | 72 s | 72 s | 72 s | 70 s | -0.2% | 34/36 | 27 / 24 |
| Flash-Next W3-C4 | 159 s | 157 s | 157 s | 157 s | 157 s | -0.1% | 99/100 | 77 / 81 |
| 27B W1 | 589 s | 596 s | 596 s | 596 s | 596 s | +0.0% | 36/36 | 39 / 64 |
| 27B W2 | 293 s | 293 s | 236 s | 236 s | 233 s | -0.0% | 32/36 | 19 / 21 |
| 27B W3 | 352 s | 347 s | 347 s | 347 s | 347 s | +0.1% | 56/60 | 42 / 48 |
| 27B W4 | 403 s | 430 s | 430 s | 394 s | 376 s | -1.3% | 27/33 | 31 / 26 |
| 27B W2-C4 | 230 s | 231 s | 231 s | 231 s | 227 s | -0.2% | 28/36 | 22 / 25 |
| 27B W3-C4 | 478 s | 497 s | 497 s | 497 s | 497 s | -0.1% | 97/100 | 92 / 97 |

| Model | Restart before request | Shutdown | Today | Phase 0 | Hybrid | Restored after restart: today / Phase 0 / hybrid |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Flash-Next | 12 | graceful | 137 s | 137 s | 137 s | 60,560 / 60,560 / 60,560 |
| Flash-Next | 12 | abrupt | 137 s | 137 s | 137 s | 60,560 / 60,560 / 60,560 |
| Flash-Next | 20 | graceful | 161 s | 137 s | 137 s | 75,275 / 104,810 / 104,810 |
| Flash-Next | 20 | abrupt | 161 s | 149 s | 137 s | 75,275 / 90,034 / 104,810 |
| Flash-Next | 28 | graceful | 185 s | 137 s | 137 s | 75,275 / 134,388 / 134,388 |
| Flash-Next | 28 | abrupt | 185 s | 149 s | 137 s | 75,275 / 119,579 / 134,388 |
| 27B | 12 | graceful | 597 s | 597 s | 597 s | 60,396 / 60,396 / 60,396 |
| 27B | 12 | abrupt | 649 s | 649 s | 597 s | 45,676 / 45,676 / 60,396 |
| 27B | 20 | graceful | 718 s | 597 s | 597 s | 75,026 / 104,386 / 104,386 |
| 27B | 20 | abrupt | 718 s | 659 s | 597 s | 75,026 / 89,697 / 104,386 |
| 27B | 28 | graceful | 855 s | 597 s | 597 s | 75,026 / 133,842 / 133,842 |
| 27B | 28 | abrupt | 855 s | 667 s | 597 s | 75,026 / 119,098 / 133,842 |

| Run | Disk written: server | Today (sim) | Phase 0 | Hybrid | Hybrid + dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash-Next W1 | 7.1 GB | 7.0 GB | 24.1 GB | 5.7 GB | 5.7 GB |
| Flash-Next W2 | 16.3 GB | 16.6 GB | 15.8 GB | 5.3 GB | 5.7 GB |
| Flash-Next W3 | 13.7 GB | 13.5 GB | 13.5 GB | 6.3 GB | 6.3 GB |
| Flash-Next W4 | 23.2 GB | 26.7 GB | 29.0 GB | 7.1 GB | 6.8 GB |
| Flash-Next W2-C4 | 14.5 GB | 15.6 GB | 15.6 GB | 5.4 GB | 5.8 GB |
| Flash-Next W3-C4 | 16.2 GB | 15.9 GB | 15.9 GB | 8.7 GB | 8.7 GB |
| 27B W1 | 16.5 GB | 16.5 GB | 56.8 GB | 13.2 GB | 13.2 GB |
| 27B W2 | 37.9 GB | 38.4 GB | 36.5 GB | 12.0 GB | 13.0 GB |
| 27B W3 | 31.7 GB | 31.5 GB | 31.5 GB | 13.5 GB | 13.4 GB |
| 27B W4 | 70.9 GB | 71.4 GB | 71.4 GB | 17.2 GB | 16.0 GB |
| 27B W2-C4 | 36.1 GB | 36.6 GB | 36.6 GB | 11.6 GB | 12.6 GB |
| 27B W3-C4 | 33.7 GB | 33.6 GB | 33.6 GB | 20.1 GB | 20.0 GB |
