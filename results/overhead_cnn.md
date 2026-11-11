### Overhead per iteration and per epoch (cnn net, 21,840 params, batch 256, 234 iterations/epoch, 8 CPU threads, idle machine, mean over 150 iterations)

| optimizer | grad evals / it | extra forwards / it | model GFLOPs / epoch | optimizer GFLOPs / epoch | ms / it: model | ms / it: optimizer | s / epoch | × SGD time |
|---|---|---|---|---|---|---|---|---|
| SGD + momentum | 1.00 | 0.00 | 155 | – | 18.2 | 0.5 | 4.4 | 1.0 |
| Adam | 1.00 | 0.00 | 155 | – | 15.6 | 0.5 | 3.8 | 0.9 |
| AdamW | 1.00 | 0.00 | 155 | – | 16.2 | 0.5 | 3.9 | 0.9 |
| Muon | 1.00 | 0.00 | 155 | – | 14.9 | 1.6 | 3.9 | 0.9 |
| Shampoo (Adam grafting, preconditioner every 10 steps) | 1.00 | 0.00 | 155 | – | 15.9 | 3.0 | 4.4 | 1.0 |
| SophiaG (Hessian estimate every 10 steps) | 1.10 | 0.00 | 171 | – | 18.3 | 0.5 | 4.4 | 1.0 |
| ARCs-LSR1, full network, m=10 | 2.00 | 0.00 | 311 | 1.2 | 31.1 | 4.3 | 8.3 | 1.9 |
| per-layer models, shared μ, m=10 | 2.00 | 0.00 | 311 | 1.2 | 46.7 | 27.4 | 17.3 | 4.0 |
| per-layer μ_l + floor, m=10 | 2.00 | 5.00 | 599 | 1.2 | 66.4 | 22.4 | 20.8 | 4.7 |
| per-layer μ_l, m=2 | 2.00 | 4.00 | 541 | 0.2 | 54.7 | 9.7 | 15.1 | 3.4 |
| per-layer μ_l, m=2, Hessian momentum | 2.00 | 4.00 | 541 | 0.5 | 56.8 | 13.4 | 16.4 | 3.8 |

Model FLOPs per sample (torch FlopCounterMode): forward 9.61e+05, forward+backward 2.6e+06. Optimizer FLOPs count only the products with parameter-sized arrays (O(n·m)); O(n) vector operations are not counted, and for SGD/Adam the optimizer cost is a few O(n) operations per step.
