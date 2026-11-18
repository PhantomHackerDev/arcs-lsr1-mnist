### Single-precision L-SR1 state vs. float64 (F-MNIST CNN, 10 epochs, 3 seeds)

| batch, m, δ rule | float64: train loss | float32: train loss | float64 / float32 test acc % | max |Δ loss| over seeds | wall time per run, float64 / float32 (s) |
|---|---|---|---|---|---|
| 256, m = 50, safe_lm + double_grad | 0.173 ± 0.002 (n=3) | 0.172 ± 0.013 (n=3) | 90.17 / 90.16 | 0.0129 | 687 / 633 |
| 1024, m = 50, safe_lm + double_grad | 0.195 ± 0.004 (n=3) | 0.197 ± 0.006 (n=3) | 90.34 / 90.24 | 0.0032 | 660 / 631 |
| 2048, m = 100, safe_lm | 0.243 ± 0.014 (n=3) | 0.242 ± 0.014 (n=3) | 89.43 / 89.70 | 0.0065 | 630 / 666 |
| 4096, m = 100, safe_keep | 0.284 ± 0.017 (n=3) | 0.283 ± 0.011 (n=3) | 88.33 / 88.40 | 0.0099 | 500 / 662 |

Wall times are from runs sharing the machine (one thread each), so they are only indicative.

