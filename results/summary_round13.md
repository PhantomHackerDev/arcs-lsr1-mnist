### Round 13: the δ rule of the authors' code (F-MNIST CNN, 10 epochs)

All four rules use δ = 0.5 λ̂_min when λ̂_min > 0; they differ only when λ̂_min ≤ 0. Training loss after 10 epochs (3 seeds unless n is shown).

| batch, m | δ rule (λ̂_min ≤ 0 case) | train loss | test acc % | accepted % | λ̂_min ≤ 0 (% of steps) | δ < 0 (% of steps) | median δ |
|---|---|---|---|---|---|---|---|
| 2048, m = 20 | safe: 0.5 λ̂_min, else δ = 1 | 0.328 | 86.95 | 83 | 18 | 0 | 0.0932 |
| 2048, m = 20 | safe_keep: else previous δ | 0.322 ± 0.023 (n=3) | 87.26 | 87 | 3 | 0 | 0.0877 |
| 2048, m = 20 | safe_lm: else previous δ + LM damping | 0.312 ± 0.005 (n=3) | 87.51 | 88 | 4 | 0 | 0.0872 |
| 2048, m = 20 | code: else δ = 1.5 λ̂_min < 0 (authors) | 0.301 ± 0.012 (n=3) | 87.91 | 89 | 3 | 3 | 0.0854 |

| 2048, m = 50 | safe: 0.5 λ̂_min, else δ = 1 | 0.408 | 84.42 | 78 | 100 | 0 | 1 |
| 2048, m = 50 | safe_keep: else previous δ | 0.259 ± 0.007 (n=3) | 88.93 | 97 | 6 | 0 | 0.0319 |
| 2048, m = 50 | safe_lm: else previous δ + LM damping | 0.258 ± 0.008 (n=3) | 89.03 | 96 | 25 | 0 | 0.0426 |
| 2048, m = 50 | code: else δ = 1.5 λ̂_min < 0 (authors) | 0.260 ± 0.006 (n=3) | 89.00 | 96 | 12 | 12 | 0.0284 |

| 2048, m = 100 | safe_keep: else previous δ | 0.305 ± 0.102 (n=3) | 87.33 | 92 | 33 | 0 | 0.0196 |
| 2048, m = 100 | safe_lm: else previous δ + LM damping | 0.243 ± 0.014 (n=3) | 89.43 | 96 | 33 | 0 | 0.0196 |
| 2048, m = 100 | code: else δ = 1.5 λ̂_min < 0 (authors) | 0.242 ± 0.015 (n=3) | 89.47 | 97 | 30 | 29 | 0.00299 |

| 2048, m = 200 | safe_keep: else previous δ | 0.477 | 80.66 | 89 | 100 | 0 | 0.292 |
| 2048, m = 200 | safe_lm: else previous δ + LM damping | 0.238 ± 0.010 (n=3) | 89.60 | 96 | 33 | 0 | 0.00569 |
| 2048, m = 200 | code: else δ = 1.5 λ̂_min < 0 (authors) | 0.245 ± 0.012 (n=3) | 89.37 | 96 | 33 | 33 | 0.00117 |

| 1024, m = 50 + double_grad | safe_keep: else previous δ | 0.237 ± 0.059 (n=3) | 89.25 | 88 | 66 | 0 | 0.0718 |
| 1024, m = 50 + double_grad | safe_lm: else previous δ + LM damping | 0.195 ± 0.004 (n=3) | 90.34 | 97 | 15 | 0 | 0.0885 |
| 1024, m = 50 + double_grad | code: else δ = 1.5 λ̂_min < 0 (authors) | 0.193 ± 0.007 (n=3) | 90.13 | 97 | 13 | 13 | 0.079 |

| 4096, m = 50 + double_grad | safe_keep: else previous δ | 0.335 ± 0.118 (n=3) | 86.23 | 82 | 33 | 0 | 0.00993 |
| 4096, m = 50 + double_grad | safe_lm: else previous δ + LM damping | 0.284 ± 0.030 (n=3) | 88.16 | 84 | 33 | 0 | 0.00993 |
| 4096, m = 50 + double_grad | code: else δ = 1.5 λ̂_min < 0 (authors) | 0.271 ± 0.009 (n=3) | 88.57 | 87 | 33 | 32 | 0.0025 |

