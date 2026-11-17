### Round 10: two ideas from Goldfarb, Ren & Bahamou (2020), F-MNIST CNN

ARCs-LSR1, full network, same-batch y, 10 epochs; train loss on the full training set after 10 epochs (test accuracy); compute in full-training-set gradient evaluations.

#### (a) LM damping in the δ rule ('safe_lm')

| batch, m, δ rule | seeds | train loss | test acc % | accepted % | λ̂_min ≤ 0 (% of steps) | compute | median LM shift when applied |
|---|---|---|---|---|---|---|---|
| 2048, m = 10, safe | 3 | 0.344 ± 0.008 (n=3) | 86.57 | 77 | 1.4 | 19.8 | – |
| 2048, m = 20, safe | 1 | 0.328 | 86.95 | 83 | 17.6 | 19.8 | – |
| 2048, m = 50, safe | 1 | 0.408 | 84.42 | 78 | 99.7 | 19.8 | – |
| 2048, m = 10, safe_keep | 3 | 0.352 ± 0.021 (n=3) | 86.23 | 77 | 1.5 | 19.8 | – |
| 2048, m = 20, safe_keep | 3 | 0.322 ± 0.023 (n=3) | 87.26 | 87 | 2.7 | 19.8 | – |
| 2048, m = 50, safe_keep | 3 | 0.259 ± 0.007 (n=3) | 88.93 | 97 | 6.1 | 19.8 | – |
| 2048, m = 100, safe_keep | 3 | 0.305 ± 0.102 (n=3) | 87.33 | 92 | 33.2 | 19.8 | – |
| 2048, m = 200, safe_keep | 1 | 0.477 | 80.66 | 89 | 99.7 | 19.8 | – |
| 2048, m = 20, safe_lm | 3 | 0.312 ± 0.005 (n=3) | 87.51 | 88 | 3.6 | 19.8 | 1.12 |
| 2048, m = 50, safe_lm | 3 | 0.258 ± 0.008 (n=3) | 89.03 | 96 | 25.0 | 19.8 | 0.678 |
| 2048, m = 100, safe_lm | 3 | 0.243 ± 0.014 (n=3) | 89.43 | 96 | 33.2 | 19.8 | 0.812 |
| 2048, m = 200, safe_lm | 3 | 0.238 ± 0.010 (n=3) | 89.60 | 96 | 33.2 | 19.8 | 1.21 |
| 1024, m = 200, safe_keep | 1 | 0.202 | 90.02 | 99 | 0.0 | 19.8 | – |
| 1024, m = 200, safe_lm | 3 | 0.215 ± 0.012 (n=3) | 89.70 | 99 | 0.0 | 19.8 | 0 |
| 4096, m = 200, safe_keep | 1 | 0.274 | 88.61 | 94 | 0.0 | 19.1 | – |
| 4096, m = 200, safe_lm | 3 | 0.282 ± 0.018 (n=3) | 88.40 | 92 | 0.0 | 19.1 | 0 |

#### (b) gradient over two batches at no extra cost ('double_grad')

m = 50, δ rule safe_keep. 'double_grad': model gradient = mean of the current and the previous batch's gradient at the current point. '+ ratio': the ratio is also measured on the two-batch mean loss (one extra forward pass per step).

| batch, variant | seeds | train loss | test acc % | accepted % | λ̂_min ≤ 0 (% of steps) | compute | geometric mean over epochs 1-10 | ratio to reference, per epoch |
|---|---|---|---|---|---|---|---|---|
| 256, reference (safe_keep) | 3 | 0.207 ± 0.003 (n=3) | 89.68 | 100 | 33.3 | 20.0 | 0.2546 | – |
| 256, double_grad (safe_keep) | 3 | 0.172 ± 0.007 (n=3) | 90.13 | 100 | 1.6 | 20.0 | 0.2195 | 0.81-0.92 |
| 256, double_grad + ratio (safe_keep) | 3 | 0.176 ± 0.004 (n=3) | 89.99 | 100 | 34.1 | 23.7 | 0.2218 | 0.82-0.94 |
| 256, reference (safe_lm) | 3 | 0.210 ± 0.008 (n=3) | 89.81 | 100 | 0.9 | 20.0 | 0.2521 | – |
| 256, double_grad (safe_lm) | 3 | 0.173 ± 0.002 (n=3) | 90.17 | 100 | 2.0 | 20.0 | 0.2199 | 0.81-0.93 |
| 1024, reference (safe_keep) | 3 | 0.238 ± 0.009 (n=3) | 89.32 | 99 | 0.0 | 19.8 | 0.2962 | – |
| 1024, double_grad (safe_keep) | 3 | 0.237 ± 0.059 (n=3) | 89.25 | 88 | 66.5 | 19.8 | 0.292 | 0.96-1.02 |
| 1024, double_grad + ratio (safe_keep) | 3 | 0.225 ± 0.044 (n=3) | 89.56 | 90 | 66.5 | 23.5 | 0.2988 | 0.93-1.05 |
| 1024, reference (safe_lm = safe_keep here) | 3 | 0.238 ± 0.009 (n=3) | 89.32 | 99 | 0.0 | 19.8 | 0.2962 | – |
| 1024, double_grad (safe_lm) | 3 | 0.195 ± 0.004 (n=3) | 90.34 | 97 | 15.5 | 19.8 | 0.2589 | 0.82-0.95 |
| 2048, reference (safe_keep) | 3 | 0.259 ± 0.007 (n=3) | 88.93 | 97 | 6.1 | 19.8 | 0.352 | – |
| 2048, double_grad (safe_keep) | 3 | 0.306 ± 0.069 (n=3) | 87.57 | 81 | 72.2 | 19.8 | 0.4035 | 1.07-1.23 |
| 2048, double_grad + ratio (safe_keep) | 3 | 0.338 ± 0.029 (n=3) | 86.66 | 79 | 99.3 | 23.5 | 0.4612 | 1.12-1.46 |
| 2048, reference (safe_lm) | 3 | 0.258 ± 0.008 (n=3) | 89.03 | 96 | 25.0 | 19.8 | 0.346 | – |
| 2048, double_grad (safe_lm) | 3 | 0.257 ± 0.040 (n=3) | 88.79 | 90 | 65.2 | 19.8 | 0.3262 | 0.90-0.99 |
| 4096, reference (safe_keep) | 3 | 0.294 ± 0.019 (n=3) | 88.07 | 93 | 0.0 | 19.1 | 0.4499 | – |
| 4096, double_grad (safe_keep) | 3 | 0.335 ± 0.118 (n=3) | 86.23 | 82 | 33.1 | 19.1 | 0.4601 | 0.83-1.10 |
| 4096, double_grad + ratio (safe_keep) | 3 | 0.308 ± 0.066 (n=3) | 87.28 | 83 | 33.1 | 22.6 | 0.4676 | 0.83-1.12 |
| 4096, reference (safe_lm = safe_keep here) | 3 | 0.294 ± 0.019 (n=3) | 88.07 | 93 | 0.0 | 19.1 | 0.4499 | – |
| 4096, double_grad (safe_lm) | 3 | 0.284 ± 0.030 (n=3) | 88.16 | 84 | 33.1 | 19.1 | 0.4219 | 0.79-0.99 |
