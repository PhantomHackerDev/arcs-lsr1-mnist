### Round 12: diagonal B0 = δ D (F-MNIST CNN, 10 epochs, 3 seeds)

D = (√v̂ + 1e-8)^p normalized to mean 1, v the EMA of g² (β₂ = 0.999), refreshed every step; p = 0 is the reference (rounds 9-10). Training loss on the full training set after 10 epochs; optimizer FLOPs relative to the model's forward/backward FLOPs.

| batch, setting | p | train loss | test acc % | accepted % | optimizer / model FLOPs | ratio to p = 0, per epoch |
|---|---|---|---|---|---|---|
| 256, m = 50, double_grad | 0 | 0.173 ± 0.002 (n=3) | 90.17 | 100 | 2.0% | – |
| 256, m = 50, double_grad | 0.5 | 0.172 ± 0.003 (n=3) | 90.10 | 100 | 34.8% | 0.98-1.03 |
| 256, m = 50, double_grad | 1 | 0.174 ± 0.012 (n=3) | 90.07 | 100 | 34.8% | 0.98-1.03 |
| 1024, m = 50, double_grad | 0 | 0.195 ± 0.004 (n=3) | 90.34 | 97 | 0.5% | – |
| 1024, m = 50, double_grad | 0.5 | 0.196 ± 0.006 (n=3) | 90.34 | 99 | 8.7% | 0.93-1.00 |
| 1024, m = 50, double_grad | 1 | 0.194 ± 0.012 (n=3) | 89.99 | 99 | 8.7% | 0.92-1.00 |
| 2048, m = 100 | 0 | 0.243 ± 0.014 (n=3) | 89.43 | 96 | 0.5% | – |
| 2048, m = 100 | 0.5 | 0.238 ± 0.010 (n=3) | 89.67 | 99 | 16.9% | 0.92-1.01 |
| 2048, m = 100 | 1 | 0.251 ± 0.030 (n=3) | 89.23 | 95 | 16.9% | 0.88-1.03 |
| 4096, m = 100 | 0 | 0.284 ± 0.017 (n=3) | 88.33 | 92 | 0.2% | – |
| 4096, m = 100 | 0.5 | 0.276 ± 0.003 (n=3) | 88.60 | 95 | 8.4% | 0.76-0.98 |
| 4096, m = 100 | 1 | 0.273 ± 0.005 (n=3) | 88.63 | 96 | 8.4% | 0.78-0.96 |

μ_min with and without the diagonal B0 (the round-9 value is μ_min = 1):

| batch, B0 | μ_min = 0.0625 | μ_min = 0.25 | μ_min = 1 | μ_min = 4 |
|---|---|---|---|---|
| 1024, δI | 0.337 ± 0.038 (n=3) | 0.228 ± 0.013 (n=3) | 0.195 ± 0.004 (n=3) | – |
| 1024, δD, p = 1 | 0.202 ± 0.013 (n=3) | 0.184 ± 0.008 (n=3) | 0.194 ± 0.012 (n=3) | 0.215 ± 0.010 (n=3) |
| 4096, δI | 0.348 ± 0.019 (n=3) | 0.331 ± 0.025 (n=3) | 0.284 ± 0.017 (n=3) | – |
| 4096, δD, p = 1 | 0.302 ± 0.026 (n=3) | 0.268 ± 0.011 (n=3) | 0.273 ± 0.005 (n=3) | 0.324 ± 0.028 (n=3) |

Best diagonal B0 vs. the tuned baselines, training loss after 10 epochs (3 seeds; baselines at their epoch-10 learning rate). In parentheses: the baselines after 20 epochs, i.e. at the same compute as 10 ARCs-LSR1 epochs (not run at batch 256).

| batch | ARCs-LSR1, diagonal B0 | AdamW | SophiaG | Shampoo | Muon |
|---|---|---|---|---|---|
| 256 | 0.172 ± 0.003 (p = 0.5) | 0.210 | 0.196 | 0.146 | 0.139 |
| 1024 | 0.184 ± 0.008 (p = 1, μ_min = 0.25) | 0.224 (0.173) | 0.233 (0.156) | 0.164 (0.126) | 0.138 (0.072) |
| 2048 | 0.238 ± 0.010 (p = 0.5) | 0.259 (0.200) | 0.261 (0.193) | 0.189 (0.128) | 0.136 (0.078) |
| 4096 | 0.268 ± 0.011 (p = 1, μ_min = 0.25) | 0.316 (0.248) | 0.338 (0.230) | 0.248 (0.166) | 0.189 (0.085) |
