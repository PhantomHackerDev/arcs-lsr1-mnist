# ARCs-LSR1 on MNIST: batch-size, γ, and the three suggested directions

Reproduction and extension of *Symmetric Rank-One Quasi-Newton Methods for Deep
Learning Using Cubic Regularization* (Ranganath, Singhal, Marcia, arXiv:2502.12298).
Everything here uses only public data (MNIST) and the public paper.

## Questions

1. **Batch size.** As mini-batches get smaller and noisier, does cubic
   regularization (μ) stabilize the SR1 step, or does the quality of the curvature
   estimate become the bottleneck?
2. **γ and RMRᵀ.** Aditya noted that most instability came from the choice of γ
   (B₀ = γI, written δ in the paper) and from the eigendecomposition of RMRᵀ.
   Where exactly does it come from?
3. **The three suggested directions:**
   - D1: a separate cubic subproblem for each layer
   - D2: a weighted momentum term
   - D3: measuring s and y on the same batch
4. **Round 2, Aditya's follow-up questions on D1:**
   - Is a separate μ_l per layer better than one shared μ? Each μ_l is adapted with a ratio computed from f with only layer l moved, so no Jacobians are needed.
   - Can a Hessian momentum B̃_{t+1} = αB̃_t + (1−α)B_new reach the same accuracy with a history of only 1–2 pairs?

## Setup

| item | value |
|---|---|
| network | the paper's MNIST classifier, 784-500-ReLU-10, 397,510 parameters (dropout removed, see note) |
| data | MNIST, standard normalisation, 60k train / 10k test |
| ARCs-LSR1 | history m = 10, μ₀ = 1, η₁ = 0.1, η₂ = 0.75, γ₁ = 1, γ₂ = 3 (μ: ×½ very successful, ×1 successful, ×2 rejected), SR1 skip ε = 1e-8, one iteration per mini-batch (the paper's batch-size study setting) |
| baselines | SGD + momentum 0.9, lr 0.1; Adam lr 1e-3, ε 1e-6 (the paper's settings) |
| grid | batch size {64, 256, 1024} × 10 ARCs configurations + 2 baselines × 3 seeds, 10 epochs each |
| diagnostics | every iteration: μ, ρ, accept, γ, eigenvalues of B, cond(M⁻¹), cond(Ψ), SR1 skips. Every 50 iterations: Hessian-vector products on a fixed 4,096-sample batch to check what B claims |

**Why dropout is removed:** the ARC ratio ρ = (f(x) − f(x+s)) / pred compares two
loss values. With dropout, those two values come from different random sub-networks,
so optimizer noise and dropout noise would be mixed together.

### Configurations

| name | γ (B₀ = γI) | how y is measured | other |
|---|---|---|---|
| `paper_cross` | γ = λ̂_min, the smallest eigenvalue of (D+L+Lᵀ)u = λ̂ SᵀS u (Algorithm 1, step 4, read literally), clipped to [1e-4, 1e4] | y = g_{B_{k+1}}(x_{k+1}) − g_{B_k}(x_k) (two different batches, the result of an `torch.optim.LBFGS`-style loop) | |
| `safe_cross` | γ = ½ λ̂_min if λ̂_min > 0, else 1 (strictly below λ̂_min, as in the text and Erway et al. 2020, Lemma 2.4) | cross-batch | |
| `bb_cross` | γ = yᵀy / sᵀy (usual L-BFGS scaling) | cross-batch | pilot only (150 iterations, 1 seed): μ hit 1.7e10 and B had an eigenvalue of −4.6e6 where the true curvature was +3.2, so it was dropped from the final grid |
| `*_same` | as above | **y = g_{B_k}(x_k + s_k) − g_{B_k}(x_k)** (same batch, D3) | the ratio test already evaluates f at x_k + s_k on this batch, so y costs one extra backward pass |
| `*_layer` | safe, per block | | **D1**: a separate L-SR1 model, γ, and shape-changing-norm subproblem for each layer (block-diagonal B); one shared μ and one global ρ test |
| `*_gb0.9` | safe | | **D2**, first interpretation: the cubic model's linear term uses an EMA of the gradient (β = 0.9, bias-corrected). Pilot only: an interrupted batch-64 run reached 79–88% after 6–7 epochs, against ~98% without it, so it was dropped. Round 2 tests the momentum Aditya actually meant, on the Hessian approximation. |

## Round 1 results (the paper's 2-layer net, 10 epochs, 3 seeds)

Full table: [results/summary.md](results/summary.md). Test accuracy %, mean ± sd:

| batch | SGD | Adam | γ = λ̂_min, cross-batch y | γ = ½λ̂_min, cross-batch y | **same-batch (s, y)** | per-layer, cross-batch | per-layer + same-batch |
|---|---|---|---|---|---|---|---|
| 64 | 96.26 ± 0.14 | 97.84 ± 0.15 | 77.08 ± 2.38 | 70.19 ± 9.23 | **98.16 ± 0.02** | 87.12 ± 1.47 | 97.94 ± 0.15 |
| 256 | 98.43 ± 0.03 | 98.13 ± 0.08 | 91.09 ± 0.60 | 79.20 ± 8.85 | **98.28 ± 0.13** | 92.84 ± 0.35 | 98.36 ± 0.03 |
| 1024 | 98.05 ± 0.07 | 97.87 ± 0.19 | 97.16 ± 0.27 | 91.14 ± 1.81 | **98.30 ± 0.10** | 88.71 ± 1.91 | 98.25 ± 0.18 |

**1. With cross-batch y, curvature quality is the bottleneck, not μ.**
- When y is the difference of gradients from two different mini-batches, ARCs-LSR1 gets worse as the batch shrinks.
- At batch 64, μ climbs to the 1e12 cap and stays there, and only about half of the steps are accepted ([fig 2](figures/fig2_mu_vs_gamma.png)).
- μ does what it is designed to do: it damps a model it cannot trust. It cannot repair the model.

**2. Measuring s and y on the same batch (D3) removes the problem.**
- Accuracy is 98.2–98.3% at every batch size, with sd ≤ 0.13.
- The median μ stays in 0.5–2 (the maximum was 27) and more than 90% of steps are accepted ([fig 1](figures/fig1_accuracy.png)).
- The ratio test already evaluates f(x + s) on the current batch, so the only extra cost is one backward pass.
- On this problem it is about even with tuned SGD and Adam, not better.

**3. The negative curvature that B reports in the cross-batch loop is not real.**
- Every 50 iterations, I took the most negative eigen-direction u of B and evaluated uᵀHu with a Hessian on a fixed 4,096-sample batch.
- Out of 1,181 such directions, across all batch sizes and both γ rules, **none** had negative curvature on the larger batch ([fig 3](figures/fig3_negative_curvature.png)).
- With same-batch pairs, B was never indefinite.

**4. The γ rule decides whether B can be indefinite at all.**
- On random instances (1,344 / 1,344), 0 < γ < λ̂_min always gave a positive definite B, and γ slightly above λ̂_min always gave an indefinite B.
- So under the rule stated in the text, negative curvature can only enter when λ̂_min ≤ 0. In the cross-batch loop that happened mostly because of noisy pairs.
- γ = λ̂_min exactly, as Algorithm 1 step 4 reads, sits on the boundary where M⁻¹ is singular. The median cond(M⁻¹) was 3.7e14 at batch 64 and 5.4e15 in the unit test.

**5. Per-layer subproblems (D1)**
- With same-batch pairs, they match the full network.
- With cross-batch pairs, they help at batch 64/256 (87% / 93% vs 70% / 79%) but not at batch 1024.
- Round 2 studies this on a deeper network, with a separate μ_l per layer.

## Round 2 results (deep net 784-512-256-128-10, 567,434 params, 10 epochs, 3 seeds)

Full tables: [results/summary_followup.md](results/summary_followup.md). All ARCs runs use γ = ½λ̂_min.

New in the optimizer:
- **μ_l per layer.** Each layer gets ρ_l = (f(x) − f(x with only layer l moved)) / pred_l, which costs one extra forward pass per layer.
  - Layers with ρ_l ≥ η₁ move together.
  - If the combined step then fails the ratio test, only the single best layer moves (its loss is already known).
  - Each layer that moved together updates μ_l with min(ρ_l, ρ_combined).
- **Hessian momentum.** B̃ ← αB̃ + (1−α)B_new, stored as γ̃I + VΛVᵀ and truncated to rank 2m by largest |eigenvalue|. Without truncation it matches the exact matrix EMA to 2.5e-15.

### Q1: one μ vs. μ_l per layer

Test accuracy %, mean ± sd:

| batch | y | one μ, full network | per-layer models, one shared μ | per-layer models, μ_l per layer | Adam | SGD |
|---|---|---|---|---|---|---|
| 64 | same-batch | 98.02 ± 0.08 | 98.20 ± 0.25 | 97.88 ± 0.14 | 98.11 ± 0.06 | 96.53 (2 seeds; the 3rd diverged at epoch 9) |
| 256 | same-batch | 98.31 ± 0.08 | 98.34 ± 0.05 | 98.35 ± 0.02 | 98.14 ± 0.06 | 98.05 ± 0.14 |
| 64 | cross-batch | 77.56 ± 12.59 | **94.11 ± 2.19** | 41.32 ± 12.16 | | |
| 256 | cross-batch | 90.38 ± 1.88 | 90.34 ± 11.58 | 58.22 ± 9.90 | | |

The SGD baseline uses the paper's lr of 0.1, which was not retuned for the deeper net.

**1. With good curvature pairs, all three versions are equivalent** (98.0–98.35%). A separate μ_l per layer gives no measurable gain.

**2. With noisy pairs, per-layer models with one shared μ are much more robust at batch 64** (77.6% → 94.1%). At batch 256 they are no better.

**3. A separate μ_l per layer fails badly with noisy pairs.**
- Layer 1's μ_l collapses in the first epoch, and its weights grow to **63×** (batch 64) / **44×** (batch 256) their initial norm.
- All μ_l then hit the 1e12 cap and training freezes. For seed 0 at batch 64, accuracy stays at 44.67% from epoch 6 to epoch 10.
- The cause: ρ_l is measured on the same mini-batch that produced the step, and the widest layer (402k parameters) can easily fit a 64-sample batch. So a harmful step passes the per-layer test.
- With one shared μ, failures in the other layers keep μ high.
- Untested ideas: measure ρ_l on the *next* mini-batch, or bound μ_l from below by μ_global / κ.

**4. Layers do need different damping** ([fig 5](figures/fig5_layer_mu.png); fig 5 shows seed 0 only).
- Across all seeds, the widest layer (784→512) has the highest typical μ_l.
- Median μ_l over training is 4–8 for layer 1 vs 2–4 for the others at batch 256, and 130–260 vs 1–64 at batch 64.
- The last layer (128→10) has rare but extreme spikes: its 90th percentile reaches 8e3 at batch 256 and the 1e12 cap at batch 64.

**5. μ saturates late at batch 64, even with same-batch pairs.**
- After accuracy plateaus, μ ratchets up to the cap and layers freeze one by one. For seed 0 with μ_l, layer 4 froze at epoch 6 and layer 3 at epoch 10.
- The cause: near convergence the single-batch ratio is mostly noise, and the μ rule doubles μ on failure but leaves it unchanged on an ordinary success.
- Accuracy was not affected within 10 epochs.

**6. Where learning happens depends on the optimizer** ([fig 6](figures/fig6_layer_movement.png), batch 256).
- The measure is ‖w_final − w_init‖ / ‖w_init‖ per layer, which is only a proxy for learning.
- Full-network ARCs: 1.3 / 1.0 / 0.88 / 2.8 (layers 1–4).
- SGD: 1.4 / 1.3 / 1.2 / 4.0.
- Both are U-shaped, so the first and last layers move more than the middle ones, but the last layer dominates.
- Adam is the opposite: 1.7 / 1.4 / 0.80 / 0.63.

### Q2: memory vs. Hessian momentum

Batch 256, per-layer μ_l, same-batch pairs ([fig 7](figures/fig7_memory.png)):

| n-vectors stored per layer | configuration | test acc % |
|---|---|---|
| 2 | m = 1 | 98.19 ± 0.09 |
| 4 | m = 2 | **98.39 ± 0.13** |
| 4 | m = 1 + Hessian momentum α = 0.5 (rank 2) | 98.11 ± 0.08 |
| 4 | m = 1 + Hessian momentum α = 0.9 (rank 2) | 98.23 ± 0.13 |
| 8 | m = 2 + Hessian momentum α = 0.5 (rank 4) | 98.31 ± 0.09 |
| 8 | m = 2 + Hessian momentum α = 0.9 (rank 4) | 98.25 ± 0.14 |
| 10 | m = 5 | 98.43 ± 0.01 |
| 20 | m = 10 | 98.35 ± 0.02 |
| – | Adam / SGD | 98.14 ± 0.06 / 98.05 ± 0.14 |

**1. Yes, a much shorter history is enough here.**
- m = 2 (4 vectors per layer) matches m = 10 (98.39% vs 98.35%).
- Even m = 1 (98.19%) is above Adam and SGD.

**2. Hessian momentum did not help.**
- At equal memory (4 vectors), plain m = 2 is better than m = 1 with momentum.
- With 8 vectors, momentum is no better than plain m = 2 with 4 vectors.
- With α = 0.9 and m = 2, μ sat at the cap in 31% of iterations on average. The blended B̃ no longer satisfies the newest secant equation, so its predictions are less accurate.

**3. Caveat.** The differences of 0.1–0.2 points are close to the seed-to-seed spread. MNIST with same-batch pairs is also an easy case for short memories; a harder problem may need longer histories.

Wall-clock times in the summary tables are **not** comparable between rows: up to 8 runs shared 16 CPU cores under varying load.

## Round 3: fixing μ_l per layer with noisy pairs (deep net, 10 epochs, 3 seeds)

Round 2 showed that a separate μ_l per layer fails with cross-batch pairs. I tested the two fixes suggested there:

- **ρ_l on the next mini-batch** (`--rho_batch next`): the per-layer and combined ratios are measured on the next batch in the epoch, while y still comes from the current batch.
- **Floor μ_l ≥ μ_g** (`--mu_floor_kappa 1`): a global μ_g is updated from the ratio of the *full* step, so failing layers count too, and no μ_l may go below μ_g. This costs one extra forward pass.
  - A first version updated μ_g only from layers that had already passed. It never activated, because a bad layer could not raise μ_g.

Test accuracy %, mean ± sd:

| configuration | cross-batch, bs 64 | cross-batch, bs 256 | same-batch, bs 64 | same-batch, bs 256 |
|---|---|---|---|---|
| one μ, full network | 77.56 ± 12.59 | 90.38 ± 1.88 | 98.02 ± 0.08 | 98.31 ± 0.08 |
| per-layer models, shared μ | 94.11 ± 2.19 | 90.34 ± 11.58 | 98.20 ± 0.25 | 98.34 ± 0.05 |
| μ_l per layer (round 2) | 41.32 ± 12.16 | 58.22 ± 9.90 | 97.88 ± 0.14 | 98.35 ± 0.02 |
| **μ_l per layer + floor μ_l ≥ μ_g** | **94.08 ± 0.88** | 89.50 ± 2.85 | 98.15 ± 0.16 | 98.29 ± 0.05 |
| μ_l per layer, ρ_l on next batch | 81.52 ± 2.72 | 89.19 ± 1.51 | 76.82 ± 1.76 | 84.93 ± 0.81 |

**1. The floor makes μ_l per layer safe.**
- With cross-batch pairs at batch 64, accuracy goes from 41% to 94%. Layer 1 moves 2.3× its initial norm instead of 64×.
- With same-batch pairs it matches the other variants. At batch 64 it also removes the late μ saturation: no iteration had a μ_l at the cap, vs 30% of iterations without the floor.
- It does not beat per-layer models with one shared μ, though it has a smaller seed-to-seed spread at batch 64 (sd 0.88 vs 2.19).
- So the extra layer-specific damping works as a safeguard, but on MNIST it doesn't improve accuracy.

**2. Measuring ρ_l on the next batch fails** (negative result).
- It prevents the layer-1 blow-up, but some μ_l sits at the 1e12 cap in 96–99% of iterations, so training mostly freezes.
- This happens even with good same-batch pairs (77% at batch 64, 85% at batch 256).
- The cause: on an independent mini-batch, f(x) − f(x + s) is dominated by sampling noise, so even good steps often fail. Each failure doubles μ.
- A larger test batch, or a gentler μ update, might help, but neither has been tested.

## Round 4: Hessian momentum favoring B_new, rank cap, secant residual, overhead

Deep net, batch 256, per-layer μ_l, same-batch pairs, 10 epochs, 3 seeds. B̃ = w_old·B̃ + w_new·B_new. Round 2 used w_new = 0.5 and 0.1. Following Aditya's suggestion, round 4 flips this to w_new = 0.7 and 0.9 and also tries a rank cap of m instead of 2m. Full table: [results/summary_followup.md](results/summary_followup.md), [fig 7](figures/fig7_memory.png).

| history m | weight on B_new | rank cap | n-vectors / layer | test acc % |
|---|---|---|---|---|
| 1 | plain L-SR1 | – | 2 | 98.19 ± 0.09 |
| 1 | 0.9 / 0.7 | 1 | 3 | 98.18 ± 0.16 / 98.19 ± 0.04 |
| 1 | 0.9 / 0.7 | 2 | 4 | 98.05 ± 0.04 / 98.14 ± 0.08 |
| **2** | **plain L-SR1** | – | **4** | **98.39 ± 0.13** |
| 2 | 0.9 / 0.7 | 2 | 6 | 98.35 ± 0.07 / 98.32 ± 0.15 |
| 2 | 0.9 / 0.7 / 0.5 / 0.1 | 4 | 8 | 98.34 / 98.38 / 98.31 / 98.25 |

**1. Favoring B_new helps slightly, but momentum never beats plain L-SR1.**
- At m = 2, accuracy rises from 98.25% (w_new = 0.1) to 98.34–98.38% (w_new = 0.7–0.9). The differences are within one sd.
- No momentum setting beats plain L-SR1 with m = 2, which stores fewer vectors.
- A rank cap of m is as good as 2m, and slightly better at m = 1.

**2. The secant equation holds only for plain SR1.** With weight w on B_new, B̃s_k − y_k = (1 − w)(B̃_old s_k − y_k), because s_k is a new direction for B̃_old. The measured residual is linear in (1 − w):

| weight on B_new | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 (plain L-SR1) |
|---|---|---|---|---|---|---|
| median ‖B̃s_k − y_k‖ / ‖y_k‖ | 0.607 | 0.44 | 0.304 | 0.184 | 0.0625 | 3.6e-14 |
| residual / (1 − w) | 0.67 | 0.63 | 0.61 | 0.61 | 0.63 | – |

So along each new step, the previous model's prediction is about 62% off, and favoring B_new reduces the error exactly in proportion.

**3. Overhead** ([results/overhead.md](results/overhead.md)). Deep net, batch 256, 8 CPU threads, measured alone on an idle machine:

| optimizer | grad evals / it | extra forwards / it | model GFLOPs / epoch | optimizer GFLOPs / epoch | s / epoch | × SGD |
|---|---|---|---|---|---|---|
| SGD + momentum | 1 | 0 | 156 | – | 0.8 | 1.0 |
| Adam | 1 | 0 | 156 | – | 1.0 | 1.2 |
| ARCs-LSR1, full network, m = 10 | 2 | 0 | 311 | 31.9 | 11.7 | 14.7 |
| per-layer μ_l + floor, m = 10 | 2 | 5 | 650 | 31.9 | 13.3 | 16.6 |
| per-layer μ_l, m = 2 | 2 | 4 | 583 | 6.4 | 7.4 | 9.3 |

- **In FLOPs, ARCs-LSR1 costs about 2.2× SGD per epoch.** That is two gradient evaluations, plus optimizer algebra at about 10% of the model FLOPs. The per-layer μ_l variants cost about 4× because of the extra forward passes.
- **In wall time it is 9–17× slower** in this implementation. The optimizer step takes 22–46 ms vs 7–11 ms for the model. That time does not scale with FLOPs (m = 2 has 1/5 of the FLOPs of m = 10 but takes half the time), so it is dominated by float64 O(n) vector operations and Python overhead.
- A float32, fused, or GPU implementation should bring the wall time much closer to the FLOP ratio.

### Training response vs. the weight on B_new

Same runs as above (m = 2, rank cap 4) plus m = 10, SGD and Adam. Full table: [results/summary_training.md](results/summary_training.md), [fig 9](figures/fig9_training_alpha.png).
- "Per-batch improvement" is f_B(x_k) − f_B(x_{k+1}), measured on the batch the step was computed on.

| method | train loss, epoch 1 | epoch 3 | epoch 10 (mean ± sd) | per-batch improvement, epoch 1 / 5 / 10 |
|---|---|---|---|---|
| m = 2, w = 0.1 | 0.275 | 0.074 | 0.0023 ± 0.0007 | 22 / 75 / 79 % |
| m = 2, w = 0.5 | 0.190 | 0.105 | 0.0027 ± 0.0008 | 26 / 76 / 82 % |
| m = 2, w = 0.7 | 0.231 | 0.077 | 0.0026 ± 0.0010 | 24 / 75 / 81 % |
| m = 2, w = 0.9 | 0.277 | 0.096 | 0.0031 ± 0.0013 | 21 / 73 / 82 % |
| m = 2, plain L-SR1 | 0.233 | 0.116 | 0.0043 ± 0.0007 | 21 / 73 / 82 % |
| SGD + momentum | 0.106 | 0.047 | 0.0106 ± 0.0022 | – |
| Adam | 0.129 | 0.048 | 0.0111 ± 0.0017 | – |

**1. Hessian momentum lowers the final training loss, but the size of w barely matters.**
- With any w < 1, the final training loss is 0.0023–0.0031, vs 0.0043 for plain m = 2. That's about 2–3 sd, which is suggestive with only 3 seeds.
- Among the w values there is no clear ordering, and the per-batch improvement is nearly identical.
- This is the opposite of test accuracy, where plain m = 2 was best.

**2. Against SGD and Adam, ARCs is slower early but finishes lower.**
- It has a higher training loss in epochs 1–5, crosses over around epochs 6–7, and ends 3–4× lower.
- Per wall time, SGD and Adam reach a training loss of 0.01 about 10× sooner.
- SGD and Adam use the paper's fixed learning rates without decay, so part of the late gap may come from their plateau.

## Round 5: F-MNIST with the paper's CNN (21,840 params, 4 layer blocks, 10 epochs, 3 seeds)

Full table: [results/summary_fmnist.md](results/summary_fmnist.md), [fig 8](figures/fig8_fmnist.png). Test accuracy %, with the final training loss in parentheses:

| method | batch 64 | batch 256 |
|---|---|---|
| SGD + momentum (lr 0.1) | 83.06 ± 0.48 (0.43) | 87.89 ± 0.73 (0.26) |
| Adam (lr 1e-3) | **90.03 ± 0.29 (0.20)** | **88.60 ± 0.12 (0.27)** |
| ARCs-LSR1, full network, m = 10 | 73.40 ± 1.62 (0.71) | 81.54 ± 2.89 (0.46) |
| per-layer models, shared μ, m = 10 | 76.39 ± 1.88 (0.60) | 81.49 ± 1.45 (0.49) |
| per-layer μ_l + floor, m = 10 | 75.67 ± 1.52 (0.61) | 83.93 ± 0.79 (0.39) |
| per-layer models, shared μ, m = 2 | 76.25 ± 2.36 (0.63) | 83.34 ± 1.69 (0.42) |
| per-layer μ_l + floor, m = 2 | 76.78 ± 3.48 (0.69) | 85.58 ± 0.34 (0.34) |
| full network, cross-batch y | 54.84 ± 6.96 (1.21) | 67.60 ± 3.46 (0.84) |

**1. On this harder problem every ARCs-LSR1 variant is behind SGD and Adam,** in training loss as well as test accuracy.
- At batch 64, the ARCs training loss stays flat around 0.6 after the first epoch.
- The best variant (μ_l per layer + floor, m = 2, batch 256) still trails Adam by 3 points.

**2. Why: each step overfits its mini-batch.**
- Per batch, ARCs looks far better than the baselines. At batch 64, a step removes 29–45% of the current batch's loss, vs 13% for SGD and 3–6% for Adam.
- The model predicts removing 70–80% (pred / f ≈ 0.7–0.8), and about half of that is realized (ρ ≈ 0.4–0.5).
- But the model, the curvature pair and the ratio test all see only the current batch. So the step is tuned to fit that batch at the cost of the others, and the full objective barely moves.
- This is the same mechanism as the per-layer μ_l failure in round 2. On MNIST the per-batch decrease was even larger (~80%), but the bigger MLP generalized anyway.

**3. Consequence for measuring training response:** the per-batch improvement can be misleading. The full training objective is the measure to compare.

**4. Cross-batch y is much worse here too,** as on MNIST.

## Round 6: modern baselines (AdamW, Muon, Shampoo, SophiaG)

Implementations: `src/baselines.py`, tested in `src/test_baselines.py`.
- **AdamW:** `torch.optim.AdamW`.
- **Muon:** Newton–Schulz orthogonalized Nesterov momentum on the hidden weight matrices, with AdamW for the output layer and biases, as the Muon authors recommend. It reproduces `torch.optim.Muon` bitwise when both run in bfloat16; float32 is used here.
- **Shampoo:** Kronecker-factored preconditioner L^{-1/4} M R^{-1/4}, with the step size per layer grafted from Adam. The preconditioner is recomputed every 10 steps.
- **SophiaG:** clipped diagonal Gauss–Newton preconditioning, with the Gauss–Newton–Bartlett estimate every 10 steps.
- All four use decoupled weight decay 0.01.

**Tuning:** each one's learning rate was chosen by the lowest final **training** loss on seed 0, so the test set was not used. Whenever the best rate was at an edge of the range tried, the sweep was extended by a factor of 3 until it was inside. Seeds 1 and 2 then used the chosen rate. Choices and every rate tried: [results/round6_selection.json](results/round6_selection.json).

Full tables: [results/summary_modern.md](results/summary_modern.md), [fig 10](figures/fig10_modern_baselines.png). Test accuracy %, with the final training loss in parentheses:

| method | MNIST, deep MLP, bs 256 | F-MNIST, CNN, bs 64 | F-MNIST, CNN, bs 256 |
|---|---|---|---|
| ARCs-LSR1, full network, m = 10 | 98.31 (0.0026) | 73.40 (0.71) | 81.54 (0.46) |
| ARCs-LSR1, per-layer μ_l, m = 2 (+ floor on F-MNIST) | 98.39 (0.0043) | 76.78 (0.69) | 85.58 (0.34) |
| AdamW | 97.92 (0.016) | 89.55 (0.20) | 89.79 (0.21) |
| **Muon** | **98.45 (0.0001)** | **90.25 (0.13)** | **90.42 (0.14)** |
| **Shampoo** | **98.47 (0.0005)** | 90.06 (0.14) | 89.82 (0.15) |
| SophiaG | 97.88 (0.014) | 89.97 (0.18) | 89.76 (0.20) |
| SGD + momentum (paper) | 98.05 (0.011) | 83.06 (0.43) | 87.89 (0.26) |
| Adam (paper) | 98.14 (0.011) | 90.03 (0.20) | 88.60 (0.27) |

**1. Against tuned modern optimizers, ARCs-LSR1 is behind everywhere.**
- On MNIST, Muon and Shampoo beat it in training loss as well as accuracy. Muon's final training loss is 26× lower.
- So the lower final training loss that ARCs showed against the paper's SGD and Adam does not survive stronger baselines.
- On F-MNIST the gap is 4–17 points of accuracy.

**2. Muon and Shampoo are the strongest baselines here,** both on the training objective and on accuracy.

**3. Fairness caveats.**
- The ARCs hyperparameters (μ₀, η₁, η₂, the γ rule) were **not** tuned per problem, while the baselines' learning rates were.
- Every result here is 10 epochs, with no learning-rate schedules.

**Overhead per epoch, relative to SGD** (idle machine, 8 threads, batch 256: [results/overhead.md](results/overhead.md), [results/overhead_cnn.md](results/overhead_cnn.md)):

| optimizer | deep MLP (567k params) | CNN (22k params) |
|---|---|---|
| AdamW / SophiaG | 1.2× / 1.3× | 0.9× / 1.0× |
| Muon / Shampoo | 4.7× / 4.5× | 0.9× / 1.0× |
| ARCs-LSR1, full network, m = 10 | 12.5× | 1.9× |
| ARCs-LSR1, per-layer μ_l, m = 2 | 8.5× | 3.4× |
| ARCs-LSR1, per-layer μ_l + floor, m = 10 | 17.9× | 4.7× |

- **On the small CNN,** the optimizer algebra is cheap, so full-network ARCs is close to its FLOP ratio of 2×. The per-layer variants pay for one extra forward pass per layer.
- **On the large MLP,** the CPU cost of the O(n·m) algebra dominates for ARCs. For Muon it's the per-step Newton–Schulz products, and for Shampoo the eigendecompositions.
- **Wall times varied by up to ~1.7× between two benchmark runs on the same machine** (compare the round-4 table). The ratios to SGD stayed within about 15%.

## Using it as a `torch.optim` optimizer

`src/arcs_optim.py` wraps the method as a `torch.optim.Optimizer`. It uses the same closure pattern as `torch.optim.LBFGS`, and `src/test_optim.py` checks that it reproduces the functional implementation used in all experiments bitwise.

```python
from arcs_optim import ARCsLSR1Optim

layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
opt = ARCsLSR1Optim([{"params": l.parameters()} for l in layers],   # one group = one block
                    per_layer=True, y_mode="same", mu_per_layer=True, mu_floor_kappa=1.0, history=2)

for xb, yb in loader:
    def closure(backward=True):          # must evaluate the SAME batch on every call
        loss = F.cross_entropy(model(xb), yb)
        if backward:
            loss.backward()
        return loss
    loss = opt.step(closure)
```

- **Closure:** it may accept `backward`. With `backward=False` it should only compute the loss, so the ratio-test evaluations don't pay for a backward pass. A plain `closure()` also works, at that extra cost.
- **Gradients:** the optimizer zeros them itself before each evaluation.
- **Blocks:** each parameter group becomes one block when `per_layer=True`.
- **Limitation:** `state_dict()` does not include the L-SR1 memory.

## Implementation notes

- `src/arcs_lsr1.py`: the optimizer. It uses the compact representation
  B = γI + ΨMΨᵀ, the partial eigendecomposition B = U_∥(Λ̂ + γI)U_∥ᵀ + γ U_⊥U_⊥ᵀ,
  and the closed-form step s* = −α*g + U_∥(α*I − C_∥)U_∥ᵀg from the paper.
  - All quasi-Newton linear algebra runs in float64.
  - [S|Y] are kept in a preallocated buffer with an incrementally updated Gram matrix.
  - The orthonormal basis of range(Ψ) is computed from the k×k matrix ΨᵀΨ instead of an
    n×k QR. The math is the same; it is about 4× faster on CPU (≈30 ms/iteration for n = 397k).
- `src/test_math.py`: checks against brute force.
  - The closed-form 1-D minimizer matches a grid search.
  - The compact form matches recursive SR1 (relative error 2e-15).
  - The secant equations hold.
  - The step is the global minimizer of the shape-changing-norm cubic model.
  - Hessian momentum matches the exact matrix EMA, and its basis stays orthonormal.
  - μ_l per layer, the next-batch ratio and the μ floor converge on a separable toy problem, and the floor holds at every iteration (16 checks in total).
- `src/run.py` runs one experiment (`--arch paper|deep`, `--per_layer`, `--mu_per_layer`, `--rho_batch current|next`, `--mu_floor_kappa`, `--hess_beta`, `--history`).
- `src/grid.py` runs the grids (`--suite main|layer|memory|fix`).
- `src/analyze.py` and `src/analyze_followup.py` build the tables and figures, and `src/overhead.py` measures the per-epoch cost.
- `src/baselines.py` has the Muon, Shampoo and SophiaG baselines; `src/sweep_round6.py` runs their learning-rate sweep; `src/analyze_training.py`, `src/analyze_fmnist.py` and `src/analyze_modern.py` build the round-4 to round-6 tables and figures.
- `src/arcs_optim.py` is the `torch.optim.Optimizer` wrapper, and `src/test_optim.py` checks it against the functional implementation.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python src/test_math.py                       # 16 sanity checks
# round 1 (paper's 2-layer net), ~2-3 h on 16 CPU cores
.venv/bin/python src/grid.py --only paper_cross,safe_cross,safe_same,safe_cross_layer,safe_same_layer,sgd,adam
.venv/bin/python src/analyze.py                         # results/summary.md, figures/fig1-4
# round 2 (deep net: per-layer mu_l, memory / Hessian momentum)
./run_followup.sh
# round 3 (fixes for mu_l per layer)
.venv/bin/python src/grid.py --suite fix --bs 64,256 --workers 12 --threads 1
# round 4 (Hessian momentum favoring B_new, rank cap, secant residual)
./run_round4.sh
.venv/bin/python src/overhead.py --threads 8                 # on an idle machine
.venv/bin/python src/test_optim.py                           # torch.optim wrapper == functional
.venv/bin/python src/analyze_training.py                     # training response vs. the weight on B_new
# round 5 (F-MNIST, the paper's CNN)
.venv/bin/python src/grid.py --suite fmnist --bs 64,256 --workers 14 --threads 1
.venv/bin/python src/analyze_fmnist.py
# round 6 (modern baselines: initial lr grid, then adaptive extension + seeds 1,2)
.venv/bin/python src/grid.py --suite modern_mnist --bs 256 --seeds 0
.venv/bin/python src/grid.py --suite modern_fmnist --bs 64,256 --seeds 0
.venv/bin/python src/sweep_round6.py
.venv/bin/python src/analyze_modern.py
.venv/bin/python src/test_baselines.py
.venv/bin/python src/overhead.py --arch cnn --data fmnist --out overhead_cnn.md
.venv/bin/python src/analyze_followup.py                # results/summary_followup.md, figures/fig5-7
```
