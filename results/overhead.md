### Overhead per iteration and per epoch (deep net, 567,434 params, batch 256, 234 iterations/epoch, 8 CPU threads, idle machine, mean over 150 iterations)

| optimizer | grad evals / it | extra forwards / it | model GFLOPs / epoch | optimizer GFLOPs / epoch | ms / it: model | ms / it: optimizer | s / epoch | × SGD time |
|---|---|---|---|---|---|---|---|---|
| SGD + momentum | 1.00 | 0.00 | 156 | – | 5.0 | 0.8 | 1.4 | 1.0 |
| Adam | 1.00 | 0.00 | 156 | – | 6.9 | 2.5 | 2.2 | 1.6 |
| AdamW | 1.00 | 0.00 | 156 | – | 5.2 | 1.9 | 1.7 | 1.2 |
| Muon | 1.00 | 0.00 | 156 | – | 4.8 | 22.5 | 6.4 | 4.7 |
| Shampoo (Adam grafting, preconditioner every 10 steps) | 1.00 | 0.00 | 156 | – | 5.1 | 20.9 | 6.1 | 4.5 |
| SophiaG (Hessian estimate every 10 steps) | 1.10 | 0.00 | 171 | – | 5.6 | 2.0 | 1.8 | 1.3 |
| ARCs-LSR1, full network, m=10 | 2.00 | 0.00 | 311 | 31.9 | 10.8 | 61.8 | 17.0 | 12.5 |
| per-layer models, shared μ, m=10 | 2.00 | 0.00 | 311 | 31.9 | 11.2 | 65.6 | 18.0 | 13.2 |
| per-layer μ_l + floor, m=10 | 2.00 | 5.00 | 650 | 31.9 | 22.8 | 81.0 | 24.3 | 17.9 |
| per-layer μ_l, m=2 | 2.00 | 4.00 | 583 | 6.4 | 15.3 | 34.1 | 11.6 | 8.5 |
| per-layer μ_l, m=2, Hessian momentum | 2.00 | 4.00 | 583 | 13.8 | 14.5 | 46.8 | 14.3 | 10.6 |

Model FLOPs per sample (torch FlopCounterMode): forward 1.13e+06, forward+backward 2.6e+06. Optimizer FLOPs count only the products with parameter-sized arrays (O(n·m)); O(n) vector operations are not counted, and for SGD/Adam the optimizer cost is a few O(n) operations per step.
