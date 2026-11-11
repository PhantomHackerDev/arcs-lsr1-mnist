### Overhead per iteration and per epoch (deep net, 567,434 params, batch 256, 234 iterations/epoch, 8 CPU threads, idle machine, mean over 150 iterations)

| optimizer | grad evals / it | extra forwards / it | model GFLOPs / epoch | optimizer GFLOPs / epoch | ms / it: model | ms / it: optimizer | s / epoch | × SGD time |
|---|---|---|---|---|---|---|---|---|
| SGD + momentum | 1.00 | 0.00 | 156 | – | 2.9 | 0.5 | 0.8 | 1.0 |
| Adam | 1.00 | 0.00 | 156 | – | 3.1 | 1.1 | 1.0 | 1.2 |
| ARCs-LSR1, full network, m=10 | 2.00 | 0.00 | 311 | 31.9 | 6.7 | 43.3 | 11.7 | 14.7 |
| per-layer models, shared μ, m=10 | 2.00 | 0.00 | 311 | 31.9 | 6.8 | 45.0 | 12.1 | 15.2 |
| per-layer μ_l + floor, m=10 | 2.00 | 5.00 | 650 | 31.9 | 10.8 | 46.1 | 13.3 | 16.6 |
| per-layer μ_l, m=2 | 2.00 | 4.00 | 583 | 6.4 | 9.4 | 22.5 | 7.4 | 9.3 |
| per-layer μ_l, m=2, Hessian momentum | 2.00 | 4.00 | 583 | 13.8 | 9.4 | 34.7 | 10.3 | 12.9 |

Model FLOPs per sample (torch FlopCounterMode): forward 1.13e+06, forward+backward 2.6e+06. Optimizer FLOPs count only the products with parameter-sized arrays (O(n·m)); O(n) vector operations are not counted, and for SGD/Adam the optimizer cost is a few O(n) operations per step.
