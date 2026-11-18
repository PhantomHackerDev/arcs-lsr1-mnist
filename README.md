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

## Round 7: fixing the per-batch overfitting (F-MNIST CNN, batch 64)

Setting: full-network ARCs-LSR1, same-batch y, m = 10. This is where round 5 had the largest gap. Three ideas were screened on seed 0, and the selected configurations got seeds 1 and 2. As with the baselines' learning rates, selection is by the final **training** loss.
- **R:** ratio test on a larger batch, made of the current 64 samples plus random other samples (256 or 1024 in total). The gradient in the model is still the current batch's.
- **Y:** curvature pairs on that larger batch as well, y = g_R(x + s) − g_R(x).
- **H:** the ARCs hyperparameters, one at a time from the defaults: η₁, η₂, μ₀, the floor μ_min, and the γ rule.

Full table, including every screened configuration: [results/summary_round7.md](results/summary_round7.md), [fig 11](figures/fig11_round7.png). Model compute is counted per sample, so passes on the larger batch are charged at their size (1.0 = one full-training-set gradient).

| method (3 seeds) | final training loss | test acc % | compute / epoch |
|---|---|---|---|
| ARCs-LSR1, defaults (round 5) | 0.713 ± 0.050 | 73.40 ± 1.62 | 2.0 |
| ratio test on 256 samples, (η₁, η₂) = (0.01, 0.25) | 0.366 ± 0.020 | 84.98 ± 0.65 | 5.0 |
| pairs + ratio test on 256 samples, (η₁, η₂) = (0.01, 0.25) | 0.371 ± 0.018 | 84.91 ± 0.25 | 9.0 |
| floor μ_min = 16 | 0.220 ± 0.010 | 88.72 ± 0.30 | 2.0 |
| floor μ_min = 16, **no curvature pairs (B = I)** | 0.213 ± 0.015 | 89.00 ± 0.42 | 2.0 |
| **floor μ_min = 64** (selected) | **0.203 ± 0.012** | **89.54 ± 0.29** | 2.0 |
| floor μ_min = 64, **no curvature pairs (B = I)** | 0.211 ± 0.007 | 89.34 ± 0.21 | 2.0 |
| Adam (paper) | 0.205 ± 0.013 | 90.03 ± 0.29 | 1.0 |
| AdamW (tuned lr) | 0.198 ± 0.009 | 89.55 ± 0.23 | 1.0 |
| Muon (tuned lr) | 0.128 ± 0.001 | 90.25 ± 0.14 | 1.0 |

**1. The ARC μ update cannot work with a ratio measured off the batch.**
- μ halves only when ρ > η₂ = 0.75, and it doubles whenever ρ < η₁.
- Measured on a larger batch, ρ has a median of 0.2–0.37 and exceeds 0.75 in only 0–3% of steps. So μ only goes up. It reaches the cap (10¹²) and training stops: the final training loss is 0.73 with 256 samples and 0.92 with 1024.
- Even for tiny steps, ρ ≈ g_Rᵀg_B / ‖g_B‖². That value is limited by the gradient's signal-to-noise ratio, however small the step.
- Lowering (η₁, η₂) to (0.01, 0.25) lets μ settle, and the loss drops from 0.71 to 0.37 with 256 samples, at 2.5× the compute. With 1024 samples μ still blows up unless η goes lower still: (0, 0.1) gives 0.47.

**2. Lowering η on the current batch makes things worse** (seed 0).
- (η₁, η₂) = (0.01, 0.25) and (0, 0.1) diverge: the loss goes back to 2.3, which is the chance level.
- Accepting more steps that fit the current batch makes the overfitting worse. Raising η₁ helps instead: 0.29 at η₁ = 0.5.
- μ₀ hardly matters (0.63–0.77, within seed noise).

**3. The floor μ_min is the hyperparameter that matters.**
- It bounds the step: with μ ≥ μ_min, ‖s‖ ≲ √(‖g‖/μ_min).
- The seed-0 sweep over μ_min = 1, 4, 16, 64, 256 gives training losses of 0.40, 0.26, 0.22, 0.20, 0.23. The best value, 64, is inside the range.
- With a floor, μ sits at μ_min in 99.9% of the steps and 99.9% of the steps are accepted. So the ratio test and the μ adaptation no longer do anything, and the method acts like a fixed step-size method.
- The floor plays the role of the baselines' learning rate, so tuning it is the fair comparison. With it, ARCs matches Adam per epoch (0.203 vs. 0.205) at twice the compute per epoch, because it takes two gradients per step. It still trails Muon (0.128).

**4. Without curvature pairs (B = I), the result is the same within seed noise:** 0.211 vs. 0.203 at μ_min = 64, and 0.213 vs. 0.220 at μ_min = 16.
- At μ_min = 64, the cubic term μ‖s‖ ≈ 8.1, while B has δ ≈ 1.6 and λ_max ≈ 18.
- The curvature changes the step only inside the range of Ψ = Y − δS, at most m = 10 dimensions (within the span of the pairs). The rest of the 21,840-dimensional step is scaled by 1/(δ + μ‖s‖), just as it would be with B = I.
- So on this problem the improvement comes from bounding the step, not from the L-SR1 curvature.

**5. Combinations add nothing** (seed 0): μ_min = 16 together with η₁ = 0.5, or with the ratio test on 256 samples, or with pairs + ratio test on 256 samples, gives 0.208–0.227.

## Round 8: does the L-SR1 curvature help when the gradient is less noisy?

Same method as round 7 (full network, same-batch y, m = 10), with and without curvature pairs, over the floor μ_min.
- Settings: F-MNIST CNN at batch 64, 256 and 1024, and the MNIST deep MLP at batch 256.
- Without pairs, B = I, so the step is a cubic-regularized gradient step.
- For each column, μ_min was chosen by the seed-0 training loss, and seeds 1 and 2 were then run at that value.

Full tables: [results/summary_round8.md](results/summary_round8.md), [fig 12](figures/fig12_curvature_value.png).

The last epoch alone is a noisy measure. On MNIST, the late-training loss jumps by up to 5× between consecutive epochs. So the table also gives the geometric mean of the training loss over epochs 1–10, and the ratio between the two curves at each epoch.

| setting (3 seeds) | μ_min: pairs / no pairs | final train loss: pairs / no pairs | test acc %: pairs / no pairs | geometric mean over epochs: pairs / no pairs | pairs ÷ no pairs, at each epoch |
|---|---|---|---|---|---|
| F-MNIST, batch 64 | 64 / 16 | 0.203 ± 0.012 / 0.213 ± 0.015 | 89.54 / 89.00 | 0.259 / 0.261 | 0.95–1.04 |
| F-MNIST, batch 256 | 4 / 4 | **0.227 ± 0.003** / 0.333 ± 0.031 | **89.09** / 85.93 | **0.284** / 0.433 | **0.61–0.70** |
| F-MNIST, batch 1024 | 4 / 1 | **0.296 ± 0.009** / 0.608 ± 0.056 | **88.21** / 76.49 | **0.396** / 0.686 | **0.49–0.72** |
| MNIST deep MLP, batch 256 | 4 / 1 | 0.0068 ± 0.0079 / 0.0068 ± 0.0032 | 98.22 / 98.19 | **0.0210** / 0.0353 | **0.48–0.69** |

**1. The L-SR1 curvature helps once the gradient is less noisy.**
- At batch 256 and 1024 on F-MNIST, and at batch 256 on MNIST, the pairs lower the training loss by roughly 30–50% at every epoch.
- At batch 64 they make no difference (round 7).
- This fits the round-7 picture: the curvature acts only in the (at most) m directions of Ψ = Y − δS, and those pairs are only useful when they are not dominated by gradient noise.

**2. On MNIST the final epoch hides this.**
- At μ_min = 4, one seed jumped from 0.0056 to 0.0159 in the last epoch, so the final values overlap.
- The curves themselves are separated at every epoch (fig 12, bottom right).

**3. The floor helps at every batch size, and the best floor gets smaller as the batch grows:** 64 at batch 64, and 4 at batch 256 and 1024.
- On F-MNIST at batch 256, μ_min = 4 lowers the training loss from 0.459 (defaults) to 0.227.
- At the larger batches the adaptive part of ARC acts again. μ rises above the floor in 2% of the steps at batch 256, 38% at batch 1024, and 45% on MNIST. At batch 1024, 15% of the steps are rejected.
- At batch 64, μ never leaves the floor.

**4. Against the baselines (10 epochs, per epoch):**
- On F-MNIST at batch 256, tuned ARCs-LSR1 (0.227, 89.09%) beats the paper's SGD (0.259) and Adam (0.269).
- It still trails the tuned AdamW (0.210), SophiaG (0.196), Shampoo (0.146) and Muon (0.139), and it costs twice as much per epoch.
- On MNIST, Muon (0.00014) and Shampoo (0.0005) remain well ahead.
- I have not tuned the baselines at batch 1024 (done in round 9).

## Round 9: larger batches (F-MNIST CNN, batch 1024 / 2048 / 4096)

Aditya's next step: larger batches, plus his point that η₂ = 0.75 (75% agreement between the model and the actual decrease) is too optimistic.
- **ARCs-LSR1** (full network, same-batch y, η₁ = 0.1, 10 epochs): the floor μ_min ∈ {none, 1, 4, 16, 64, 256} × η₂ ∈ {0.75, 0.5, 0.25}, with and without curvature pairs (B = I).
- **Baselines** (20 epochs, so they can be compared at the same epoch and at the same compute; ARCs-LSR1 uses two gradients per step): SGD+momentum, AdamW, Muon, Shampoo, SophiaG with a learning-rate sweep extended until the best value is interior, and the paper's Adam (lr 1e-3). The lr is chosen separately at epoch 10 and at epoch 20.
- Everything is selected by the seed-0 final **training** loss; the selected configurations have 3 seeds.
- Then the L-SR1 memory m ∈ {5, 10, 20, 50, 100} at the selected (η₂, μ_min), 3 seeds.

Full tables: [results/summary_round9.md](results/summary_round9.md), [fig 13](figures/fig13_large_batch.png) (per epoch and per compute), [fig 14](figures/fig14_memory.png) (memory m). Selections: [results/round9_selection.json](results/round9_selection.json).

Training loss on the full training set (3 seeds); test accuracy in parentheses. "Epoch 10" compares the same number of samples and steps. "20 grads" compares the same compute (about 20 full-training-set gradients: 10 epochs for ARCs-LSR1, 20 for the baselines).

| method | batch 1024: epoch 10 | 20 grads | batch 2048: epoch 10 | 20 grads | batch 4096: epoch 10 | 20 grads |
|---|---|---|---|---|---|---|
| ARCs-LSR1, m = 10 (paper) | 0.328 ± 0.047 (86.8) | same | 0.344 ± 0.008 (86.6) | same | 0.405 ± 0.009 (84.3) | same |
| ARCs-LSR1, m = 50, γ rule safe_keep | 0.238 ± 0.009 (89.3) | same | **0.259 ± 0.007** (88.9) | same | 0.294 ± 0.019 (88.1) | same |
| ARCs-LSR1, m = 100, γ rule safe_keep | **0.225 ± 0.010** (89.8) | same | 0.305 ± 0.102 (87.3) | same | **0.284 ± 0.017** (88.3) | same |
| ARCs, no pairs (B = I) | 0.588 ± 0.046 (77.2) | same | 0.695 ± 0.079 (73.2) | same | 0.775 ± 0.073 (70.8) | same |
| SGD+momentum (tuned) | 0.276 (88.2) | 0.230 | 0.337 (86.4) | 0.273 | 0.427 (83.4) | 0.324 |
| Adam (paper, lr 1e-3) | 0.349 (86.5) | 0.285 | 0.405 (84.6) | 0.331 | 0.484 (81.3) | 0.394 |
| AdamW (tuned) | 0.224 (89.3) | 0.173 | 0.259 (88.7) | 0.200 | 0.316 (87.3) | 0.248 |
| SophiaG (tuned) | 0.233 (89.0) | 0.156 | 0.261 (88.5) | 0.193 | 0.338 (86.5) | 0.230 |
| Shampoo (tuned) | 0.164 (89.6) | 0.126 | 0.189 (89.8) | 0.128 | 0.248 (89.1) | 0.166 |
| Muon (tuned) | 0.138 (90.4) | 0.072 | 0.136 (89.6) | 0.078 | 0.189 (89.5) | 0.085 |

**1. Lowering η₂ makes no consistent difference at these batch sizes.**
- The best seed-0 loss for η₂ = 0.75 / 0.5 / 0.25 is 0.305 / 0.288 / 0.295 at batch 1024, 0.351 / 0.373 / 0.368 at batch 2048 and 0.406 / 0.406 / 0.425 at batch 4096. Those differences are within the spread between seeds.
- The seed-0 selection is itself noisy: at batch 1024 it picked η₂ = 0.5, μ_min = 1 (seed 0: 0.288), which over 3 seeds gives 0.328 ± 0.047, while round 8's η₂ = 0.75, μ_min = 4 gives 0.296 ± 0.009.
- The best floor is μ_min = 1 at every batch size. μ is above it in 39–96% of the steps, and 23–34% of the steps are rejected (m = 10), so the ARC adaptation is fully active.

**2. The curvature pairs halve the training loss at every batch size**: 0.33 vs. 0.59, 0.34 vs. 0.70 and 0.41 vs. 0.78 against B = I at its own best setting.

**3. With the paper's m = 10, ARCs-LSR1 is only at the level of tuned SGD. The memory is the bottleneck.**
- B differs from δI only in the range of Ψ = Y − δS, at most m = 10 of the 21,840 dimensions. In all other directions the step is the gradient scaled by 1/(δ + μ‖s‖). With δ ≈ 0.2 and μ‖s‖ ≈ 1 this is roughly the step of SGD with lr 0.1 and momentum 0.9 (effective lr 1).
- More memory helps steadily (fig 14): from m = 10 to m = 100 the loss drops from 0.328 to 0.225 at batch 1024 and from 0.405 to 0.284 at batch 4096. Accepted steps go from 66–77% to 92–99%. m = 5 is much worse (0.42–0.55).
- The cost is small at this model size: the optimizer's own arithmetic is at most 1% of the model's forward/backward FLOPs at m = 100 (0.1% at m = 10).

**4. The paper's γ rule breaks down with many pairs.**
- The 'safe' rule sets γ = λ̂_min / 2, where λ̂_min is the smallest generalized eigenvalue of (D + L + Lᵀ, SᵀS), and falls back to γ = 1 when λ̂_min ≤ 0.
- At batch 2048, λ̂_min ≤ 0 in 1.4% of the steps at m = 10, 18% at m = 20 and 99.7% at m = 50. The m = 50 run then trains with γ = 1 throughout and is worse than m = 10 (0.408).
- Keeping the previous γ instead ('safe_keep', `--delta_mode safe_keep`; identical to 'safe' whenever λ̂_min > 0) fixes m = 50 (0.259 ± 0.007).
- At m = 100 the problem returns: λ̂_min ≤ 0 in 33% of the steps and one of three seeds is much worse (0.305 ± 0.102). How to choose γ, or how to handle pairs with negative curvature, when m is large is still open.

**5. Against the tuned baselines:**
- **Per epoch**, ARCs-LSR1 with m = 50–100 matches AdamW and SophiaG at batch 1024 and 2048 and beats them at batch 4096 (0.284 vs. 0.316 and 0.338). It beats tuned SGD and the paper's Adam everywhere, but stays behind Shampoo and Muon.
- **At equal compute**, AdamW, SophiaG, Shampoo and Muon are all ahead, because each ARCs-LSR1 step costs two gradients. ARCs-LSR1 matches SGD at batch 1024 (0.225 vs. 0.230) and beats it at batch 4096 (0.284 vs. 0.324).
- Shampoo and Muon precondition every direction through the per-layer (Kronecker) structure, while ARCs-LSR1 only corrects m directions. A non-scalar initial matrix B₀ (diagonal, or per-layer Kronecker factors as in K-BFGS, Goldfarb, Ren and Bahamou 2020), with the L-SR1 correction on top, would address this directly. The cubic term would then be measured in the corresponding scaled norm.

## Round 10: two ideas from K-BFGS (Goldfarb, Ren & Bahamou, NeurIPS 2020)

Same setting as round 9: F-MNIST CNN, full network, same-batch y, 10 epochs, 3 seeds. Full tables: [results/summary_round10.md](results/summary_round10.md), [fig 15](figures/fig15_round10.png).

**(a) LM damping in the δ rule (`--delta_mode safe_lm`).**
- K-BFGS keeps its BFGS factors well conditioned with a Levenberg–Marquardt-type damping, y ← y + λs. SR1 needs no positive definiteness, but this damping has two exact properties that matter for the δ rule (both checked in `src/test_math.py`):
  - it shifts every generalized eigenvalue of (D + L + Lᵀ, SᵀS) by λ: λ̂_min(S, Y + λS) = λ̂_min(S, Y) + λ;
  - SR1 is shift invariant: the pairs (s, y + λs) with B₀ = (δ + λ)I give B + λI.
- 'safe_lm' is 'safe' whenever λ̂_min > 0. When λ̂_min ≤ 0, it keeps δ = δ_prev and damps the pairs with λ = δ_prev/0.5 − λ̂_min, so δ is again below the shifted λ̂_min. Then B = SR1(B₀ = (δ − λ)I) + λI ⪰ δI: the directions outside range(Ψ) are unchanged, and the negative curvature in range(Ψ) is shifted up.

| batch 2048, training loss after 10 epochs | m = 20 | m = 50 | m = 100 | m = 200 |
|---|---|---|---|---|
| safe (paper; seed 0) | 0.328 | 0.408 | – | – |
| safe_keep | 0.322 ± 0.023 | 0.259 ± 0.007 | 0.305 ± 0.102 | 0.477 (seed 0) |
| **safe_lm** | **0.312 ± 0.005** | **0.258 ± 0.008** | **0.243 ± 0.014** | **0.238 ± 0.010** |

- With the damping, more memory keeps helping at batch 2048, up to m = 100, and m = 200 is about the same. Without it, the large-m runs break down (λ̂_min ≤ 0 in 33–99.7% of the steps).
- At batch 1024 and 4096, λ̂_min stays positive, so 'safe_lm' = 'safe_keep' = 'safe'. There, m = 200 gives 0.215 ± 0.012 and 0.282 ± 0.018 (m = 100: 0.225 and 0.284).

**(b) Model gradient over two batches at no extra cost (`--double_grad`).**
- K-BFGS (their App. D.6) reuses the gradient from its second forward/backward pass. ARCs-LSR1 can do the same: the same-batch pair already computes g_{B_{k−1}}(x_{k−1} + s_{k−1}), which after an accepted step is the previous batch's gradient **at the current point** (after a rejected step, g_{B_{k−1}}(x_{k−1}) is). The model gradient becomes the mean of the two batches at the same point: an unbiased gradient over twice the samples, not a stale average like momentum.
- The ratio and the curvature pairs stay on the current batch. `--double_grad_ratio` also measures the ratio on the two-batch mean loss (one extra forward pass, +18% compute).

| m = 50, training loss after 10 epochs | batch 256 | batch 1024 | batch 2048 | batch 4096 |
|---|---|---|---|---|
| reference (safe_lm; = safe_keep at 1024 and 4096, where λ̂_min > 0 throughout) | 0.210 ± 0.008 | 0.238 ± 0.009 | 0.258 ± 0.008 | 0.294 ± 0.019 |
| **double_grad (safe_lm)** | **0.173 ± 0.002** | **0.195 ± 0.004** | 0.257 ± 0.040 | 0.284 ± 0.030 |
| ratio double_grad / reference, per epoch | 0.81–0.93 | 0.82–0.95 | 0.90–0.99 | 0.79–0.99 |
| double_grad (safe_keep) | 0.172 ± 0.007 | 0.237 ± 0.059 | 0.306 ± 0.069 | 0.335 ± 0.118 |
| double_grad + ratio (safe_keep) | 0.176 ± 0.004 | 0.225 ± 0.044 | 0.338 ± 0.029 | 0.308 ± 0.066 |

1. **With safe_lm, double_grad lowers the training loss at every epoch and every batch size, at no extra cost.** The gain is 5–19% at every epoch at batch 256 and 1024. At batch 2048 and 4096 the final values overlap because of one seed each (see 3).
2. **With safe_keep, double_grad is unstable at batch ≥ 1024.** In the bad seeds λ̂_min ≤ 0 from the first iterations onward, so 'safe_keep' freezes δ at its first value (0.3–0.46). The seeds where λ̂_min stays positive improve (batch 4096: 0.263 and 0.271 vs. 0.294).
3. **What is still open:** with safe_lm, one seed at batch 2048 and one at 4096 have λ̂_min ≤ 0 in 99% of the steps. δ is then never re-estimated, and those seeds end at 0.30 and 0.32 (others: 0.23–0.27). With λ̂_min persistently ≤ 0, δ needs another estimate.
4. Measuring the ratio on the two-batch mean as well gives no further gain for +18% compute.

**Best ARCs-LSR1 so far vs. the tuned baselines (training loss after 10 epochs, per epoch):**

| batch | best ARCs-LSR1 | AdamW | SophiaG | Shampoo | Muon |
|---|---|---|---|---|---|
| 256 | 0.173 (m = 50, double_grad) | 0.210 | 0.196 | 0.146 | 0.139 |
| 1024 | 0.195 (m = 50, double_grad) | 0.224 | 0.233 | 0.164 | 0.138 |
| 2048 | 0.238 (m = 200) | 0.259 | 0.261 | 0.189 | 0.136 |
| 4096 | 0.282 (m = 200) | 0.316 | 0.338 | 0.248 | 0.189 |

- Per epoch, ARCs-LSR1 now beats the tuned AdamW and SophiaG at every batch size from 256 to 4096 (at batch 2048 by a margin within the spread), but it still trails Shampoo and Muon.
- At equal compute (each step costs two gradients), the round-9 picture is unchanged. For example, at batch 1024 AdamW reaches 0.173 and SophiaG 0.156 after 20 epochs.
- m ≥ 100 combined with double_grad has not been tried yet.

## Round 11: the paper's δ rule (Aditya's question)

Aditya asked whether δ is set as in the paper, δ = λ̂_min, the smallest eigenvalue of the generalized eigenvalue problem (D + L + Lᵀ)u = λ̂ SᵀS u. Since round 2 all runs used δ = ½λ̂_min instead.
- **Why ½:** M⁻¹ = (D + L + Lᵀ) − δSᵀS is singular at δ = λ̂_min by the definition of λ̂_min (round 1, point 4). Any 0 < δ < λ̂_min keeps B positive definite (Erway et al. 2020), and ½ is one choice in that interval.
- **Setup:** round 9's setting (F-MNIST CNN, same-batch y, (η₂, μ_min) per batch size), batch 1024 / 2048 / 4096, m ∈ {10, 50, 100}, 3 seeds.
  - 'paper' is δ = λ̂_min clipped to [1e-4, 1e4], so δ = 1e-4 when λ̂_min ≤ 1e-4.
  - The fractions c·λ̂_min with c ∈ {0.9, 0.5, 0.25, 0.1} use the LM damping of round 10 when λ̂_min ≤ 0. c = 0.25 and 0.1 were run at m = 10 and 50 only.

Full table: [results/summary_round11.md](results/summary_round11.md), [fig 16](figures/fig16_delta_rule.png). Training loss after 10 epochs:

| batch, m | δ = λ̂_min (paper) | 0.9 λ̂_min | 0.5 λ̂_min | 0.25 λ̂_min | 0.1 λ̂_min |
|---|---|---|---|---|---|
| 1024, m = 10 | 0.360 ± 0.016 | 0.346 ± 0.016 | 0.328 ± 0.047 | **0.304 ± 0.008** | 0.309 ± 0.010 |
| 2048, m = 10 | 0.472 ± 0.031 | 0.433 ± 0.011 | 0.351 ± 0.019 | **0.332 ± 0.026** | **0.332 ± 0.021** |
| 4096, m = 10 | 0.538 ± 0.039 | 0.522 ± 0.024 | 0.405 ± 0.009 | **0.393 ± 0.021** | 0.400 ± 0.026 |
| 1024, m = 50 | 0.254 ± 0.015 | 0.241 ± 0.012 | 0.238 ± 0.009 | **0.231 ± 0.004** | 0.235 ± 0.010 |
| 2048, m = 50 | 0.354 ± 0.064 | 0.284 ± 0.014 | 0.258 ± 0.008 | 0.258 ± 0.007 | **0.257 ± 0.008** |
| 4096, m = 50 | 0.343 ± 0.015 | 0.309 ± 0.010 | **0.294 ± 0.019** | 0.312 ± 0.017 | 0.310 ± 0.021 |
| 1024, m = 100 | **0.222 ± 0.012** | 0.225 ± 0.015 | 0.225 ± 0.010 | – | – |
| 2048, m = 100 | 0.298 ± 0.064 | 0.258 ± 0.019 | **0.243 ± 0.014** | – | – |
| 4096, m = 100 | 0.297 ± 0.006 | 0.287 ± 0.015 | **0.284 ± 0.017** | – | – |

**1. δ = λ̂_min, as written in Algorithm 1, is the worst choice in 8 of the 9 settings.**
- At m = 10 the decomposition is numerically singular, as expected: the median cond(M⁻¹) is 5e12–1e13, against 30–70 at ½λ̂_min.
- At m ≥ 50, λ̂_min ≤ 0 in 65–99% of the steps, so δ sits at the 1e-4 clip most of the time.
- The exception is batch 1024, m = 100 (0.222 vs. 0.225), where the values are within noise.

**2. Close to λ̂_min is worse; anywhere in 0.1–0.5 is about the same.**
- 0.9λ̂_min is clearly worse than 0.5λ̂_min at m = 10 (by 5–29%) and at batch 2048 and 4096 for m = 50.
- Between 0.1 and 0.5 the differences are within the spread between seeds. At m = 10, 0.25 is slightly better than 0.5 at all three batch sizes (by 3–7%); at m = 50 there is no consistent order.
- **Recommended rule:** δ = c·λ̂_min with c ≈ 0.25–0.5, i.e. strictly inside (0, λ̂_min) with a margin, plus the LM damping of round 10 when λ̂_min ≤ 0.

## GPU and single precision (preparation for the large experiments)

- **Device.** `ARCsLSR1(..., device=...)` keeps the optimizer state on that device; `ARCsLSR1Optim` uses the parameters' device by default. `src/run.py --device cuda` (or `auto`) moves the data, the model and the optimizer state, and also works for the baselines.
- **Single precision.** `vec_dtype=torch.float32` (`--vec_dtype float32`) stores the [S | Y] buffer and all parameter-sized vectors (g, s, y) in float32. The 2m × 2m Gram matrix and all small linear algebra (eigendecompositions, λ̂_min, the cubic solution in the eigenbasis) stay in float64, and new Gram columns are accumulated in float64 in row chunks. The default (float64, CPU) reproduces every earlier result bitwise (checked against stored runs of ARCs-LSR1 and Muon).
- **Tests** (`src/test_math.py`, section 13):
  - the chunked float64 Gram equals the float64 Gram of the stored float32 values;
  - B v and the cubic step agree with float64 to single precision (relative error ~1e-7);
  - a 25-step trajectory agrees with float64.
  - A cuda-vs-cpu check runs automatically on a machine with a GPU. **It has not been run yet: this machine has no GPU.**
- **Training check** ([results/summary_fp32.md](results/summary_fp32.md)): the four best configurations of rounds 9–10, 3 seeds each. float32 and float64 agree within the spread between seeds; individual seeds differ by up to 0.013 because the trajectories diverge.

| F-MNIST, training loss after 10 epochs | float64 | float32 |
|---|---|---|
| batch 256, m = 50, safe_lm + double_grad | 0.173 ± 0.002 | 0.172 ± 0.013 |
| batch 1024, m = 50, safe_lm + double_grad | 0.195 ± 0.004 | 0.197 ± 0.006 |
| batch 2048, m = 100, safe_lm | 0.243 ± 0.014 | 0.242 ± 0.014 |
| batch 4096, m = 100 | 0.284 ± 0.017 | 0.283 ± 0.011 |

- **Memory.** The L-SR1 buffer holds 2m × n values, so it dominates at scale:

| model (parameters) | m = 10, float32 | m = 50, float32 | m = 50, float64 | m = 100, float32 |
|---|---|---|---|---|
| ResNet-18, CIFAR-10 (11.2M) | 0.9 GB | 4.5 GB | 9.0 GB | 9.0 GB |
| ResNet-50, ImageNet (25.6M) | 2.0 GB | 10.2 GB | 20.5 GB | 20.5 GB |
| GPT-2 small (124M) | 9.9 GB | 49.6 GB | 99.2 GB | 99.2 GB |

  For GPT-sized models, m = 50–100 does not fit next to the model on one GPU, so those runs need a small m, a per-layer split of the memory across devices, or a structured B₀ that needs fewer pairs.
- **GPU experiments:** [examples/README.md](examples/README.md) has the step-by-step instructions (setup, pre-flight checks, sweep, analysis, sending results back). The first package is CIFAR-10 / ResNet-18:
  - ARCs-LSR1 (through `ARCsLSR1Optim`) vs. AdamW, SGD and Muon;
  - Muon for `nn.Module`s with conv weights, checked to match the F-MNIST version exactly;
  - the same tuning protocol as rounds 9–12, spread over GPUs.
  - On the development machine (no GPU) the whole pipeline was run end to end only on CPU with random data.

## Round 12: a diagonal initial matrix B₀ = δD

Round 9 traced the gap to Shampoo and Muon to the directions outside the L-SR1 subspace, where B₀ = δI turns the step into a scaled gradient step. The first non-scalar B₀ is diagonal.

- **Method** (`--diag_power p`): D = (√v̂ + ε)^p, normalized to mean 1, where v is an EMA of g² (β₂ = 0.999, as in Adam).
  - Running ARCs-LSR1 unchanged in the variables x̃ = D^{1/2}x, on g̃ = D^{−1/2}g and the pairs (D^{1/2}s, D^{−1/2}y), is exactly L-SR1 with B₀ = δD, with the cubic term measured in the D-norm. The closed-form shape-changing-norm solution and the δ rule therefore apply as they are.
  - D is refreshed every step (`--diag_every`). The stored pairs are then rescaled exactly (s̃ ← r s̃, ỹ ← ỹ/r with r = √(D_new/D_old)) and the Gram matrix is recomputed, at O(n m²).
  - p = 0 is the method of rounds 9–11; p = 1 is an Adam-like diagonal.
  - Tests (`src/test_math.py`, section 14):
    - with D fixed, 15 steps equal plain ARCs-LSR1 on f(D^{−1/2}x̃);
    - after a refresh, the memory equals the raw pairs in the new variables;
    - B equals the compact L-SR1 with B₀ = δD.
- **Setup:** F-MNIST CNN, the best configuration of rounds 9–10 at each batch size (δ rule safe_lm; m = 50 + double_grad at batch 256 and 1024, m = 100 at batch 2048 and 4096), p ∈ {0.5, 1}, 3 seeds. At batch 1024 and 4096, μ_min was also retuned for both B₀. Full tables: [results/summary_round12.md](results/summary_round12.md), [fig 17](figures/fig17_diag_b0.png).

| training loss after 10 epochs | δI (reference) | δD, p = 0.5 | δD, p = 1 |
|---|---|---|---|
| batch 256 (μ_min = 4) | 0.173 ± 0.002 | 0.172 ± 0.003 | 0.174 ± 0.012 |
| batch 1024 (μ_min = 1) | 0.195 ± 0.004 | 0.196 ± 0.006 | 0.194 ± 0.012 |
| batch 2048 (μ_min = 1) | 0.243 ± 0.014 | 0.238 ± 0.010 | 0.251 ± 0.030 |
| batch 4096 (μ_min = 1) | 0.284 ± 0.017 | 0.276 ± 0.003 | 0.273 ± 0.005 |

| μ_min (p = 1 for δD) | 0.0625 | 0.25 | 1 | 4 |
|---|---|---|---|---|
| batch 1024, δI | 0.337 ± 0.038 | 0.228 ± 0.013 | **0.195 ± 0.004** | – |
| batch 1024, δD | 0.202 ± 0.013 | **0.184 ± 0.008** | 0.194 ± 0.012 | 0.215 ± 0.010 |
| batch 4096, δI | 0.348 ± 0.019 | 0.331 ± 0.025 | **0.284 ± 0.017** | – |
| batch 4096, δD | 0.302 ± 0.026 | **0.268 ± 0.011** | 0.273 ± 0.005 | 0.324 ± 0.028 |

**1. The diagonal B₀ helps a little, and only after retuning the floor.**
- At the round-9 floor the change is within noise, except at batch 4096 (0.284 → 0.273, with a smaller spread).
- With D, a lower floor works: μ_min = 0.25 is best for δD, while δI gets clearly worse there.
- Each at its best floor, δD is about 6% better: 0.195 → 0.184 at batch 1024 and 0.284 → 0.268 at batch 4096. The early epochs gain the most (per-epoch ratio down to 0.76 at batch 4096).
- Batch 256 and 2048 were not retuned.

**2. It does not close the gap to Shampoo and Muon.**
- At batch 4096 the best ARCs-LSR1 (0.268) is now close to Shampoo (0.248), but still behind Muon (0.189). At batch 1024: 0.184 vs. 0.164 and 0.138.
- This is expected. With a diagonal B₀, the step outside the L-SR1 subspace becomes a diagonally scaled gradient step without momentum, similar to RMSprop. Shampoo and Muon use the matrix structure of each layer, and Adam (diagonal) is also behind them. The next step is a per-layer Kronecker-factored B₀.

**3. Cost.** Refreshing D every step recomputes the Gram matrix, so the optimizer's arithmetic grows to 8–35% of the model FLOPs (0.2–2% without D). A refresh every 10–50 steps would bring this down, but has not been tested.

## Round 13: the δ rule of the authors' code

Aditya shared the δ rule of his implementation (`--delta_mode code`): δ = max(0.5 λ̂_min, 1e-6) if λ̂_min > 0, else δ = min(1.5 λ̂_min, −1e-6).
- When λ̂_min > 0 this is the 'safe' rule used since round 2. Every run here has effectively used the authors' rule in that case; Algorithm 1 in the paper writes δ = λ̂_min, which round 11 found to be the worst choice.
- When λ̂_min ≤ 0, it takes a **negative** δ below λ̂_min. The middle matrix M stays definite, while B₀ = δI is negative definite and the cubic term controls the step.
- This is compared with the three alternatives tried so far: reset to δ = 1 ('safe'), keep the previous δ ('safe_keep'), and keep it while LM-damping the pairs ('safe_lm', round 10). The settings are those where λ̂_min ≤ 0 actually happens. F-MNIST CNN, 10 epochs, 3 seeds; full table: [results/summary_round13.md](results/summary_round13.md).

| training loss after 10 epochs | safe (δ = 1) | safe_keep | safe_lm | **code (authors)** | λ̂_min ≤ 0 (% of steps, code) |
|---|---|---|---|---|---|
| batch 2048, m = 20 | 0.328 (seed 0) | 0.322 ± 0.023 | 0.312 ± 0.005 | **0.301 ± 0.012** | 3 |
| batch 2048, m = 50 | 0.408 (seed 0) | 0.259 ± 0.007 | **0.258 ± 0.008** | 0.260 ± 0.006 | 12 |
| batch 2048, m = 100 | – | 0.305 ± 0.102 | 0.243 ± 0.014 | **0.242 ± 0.015** | 30 |
| batch 2048, m = 200 | – | 0.477 (seed 0) | **0.238 ± 0.010** | 0.245 ± 0.012 | 33 |
| batch 1024, m = 50 + double_grad | – | 0.237 ± 0.059 | 0.195 ± 0.004 | **0.193 ± 0.007** | 13 |
| batch 4096, m = 50 + double_grad | – | 0.335 ± 0.118 | 0.284 ± 0.030 | **0.271 ± 0.009** | 33 |

**1. The authors' rule is as good as the LM damping, and simpler.** The two are within the spread between seeds everywhere except batch 4096 with double_grad, where the authors' rule is better and much less variable (0.271 ± 0.009 vs. 0.284 ± 0.030). Both are far better than resetting δ to 1 or only keeping it.

**2. It also fixes the case round 10 left open.** When λ̂_min stays ≤ 0 for most of the run, 'safe_lm' never re-estimates δ, and one seed out of three degraded. The authors' rule takes δ from the current λ̂_min at every step, so it adapts.

**3. Recommended rule:** the authors' code rule. In both cases it keeps δ strictly below λ̂_min with a margin (0.5 λ̂_min above zero, 1.5 λ̂_min below). Round 11 showed that the margin matters and that its exact size does not. The paper's Algorithm 1 should state this rule rather than δ = λ̂_min.

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
.venv/bin/python src/test_math.py                       # 40 sanity checks (+1 that needs a GPU)
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
# round 7 (F-MNIST batch 64: larger ratio/curvature batch, ARCs hyperparameters, no-pairs ablation)
.venv/bin/python src/grid.py --suite round7 --bs 64 --seeds 0 --workers 12 --threads 1
.venv/bin/python src/grid.py --suite round7 --bs 64 --seeds 1,2 --workers 12 --threads 1 \
    --only r7_mumin_64,r7_mumin_16,r7_nopairs_mumin_16,r7_nopairs_mumin_64,r7_rho256_eta0.01-0.25,r7_yrho256_eta0.01-0.25
.venv/bin/python src/analyze_round7.py
# round 8 (value of the L-SR1 curvature vs. batch size: mu_min sweep with / without pairs)
.venv/bin/python src/grid.py --suite round8_fm --bs 256,1024 --seeds 0 --workers 6 --threads 1
.venv/bin/python src/grid.py --suite round8_deep --bs 256 --seeds 0 --workers 6 --threads 1
.venv/bin/python src/grid.py --suite round8_fm --bs 64 --seeds 0 --only r8_fm_nopairs_mumin0,r8_fm_nopairs_mumin1,r8_fm_nopairs_mumin256
.venv/bin/python src/grid.py --suite fmnist --only fm_full_same --bs 1024 --seeds 0
.venv/bin/python src/grid.py --suite round8_fm --bs 256,1024 --seeds 1,2 --only r8_fm_pairs_mumin4,r8_fm_nopairs_mumin4,r8_fm_nopairs_mumin1
.venv/bin/python src/grid.py --suite round8_deep --bs 256 --seeds 1,2 --only r8_deep_pairs_mumin4,r8_deep_nopairs_mumin1
.venv/bin/python src/analyze_round8.py
# round 9 (F-MNIST batch 1024/2048/4096: tuned baselines over 20 epochs, ARCs mu_min x eta2 with / without pairs,
# then the L-SR1 memory m and the delta rule 'safe_keep'), ~4 h on 16 CPU cores
.venv/bin/python src/sweep_round9.py --workers 13            # -> results/round9_selection.json
.venv/bin/python src/sweep_round9m.py --workers 13
.venv/bin/python src/analyze_round9.py
# round 10 (K-BFGS ideas: LM damping in the delta rule 'safe_lm', model gradient over two batches 'double_grad')
.venv/bin/python src/sweep_round10.py --workers 13
.venv/bin/python src/analyze_round10.py
# round 11 (the paper's delta rule delta = lambda_hat_min vs. c * lambda_hat_min, c = 0.9 / 0.5 / 0.25 / 0.1)
.venv/bin/python src/sweep_round11.py --workers 13
.venv/bin/python src/analyze_round11.py
# GPU / single precision: float32 vs. float64 on the best round-9/10 configurations; CIFAR-10 template
.venv/bin/python src/check_fp32.py --workers 12
.venv/bin/python examples/test_examples.py --device cpu                 # pre-flight checks (use --device cuda on a GPU)
.venv/bin/python examples/sweep_cifar10.py --smoke --gpus cpu --per_gpu 6 && .venv/bin/python examples/analyze_cifar10.py --smoke
# round 12 (diagonal B0 = delta * D, --diag_power; mu_min retuned at batch 1024 / 4096)
.venv/bin/python src/sweep_round12.py --workers 13
.venv/bin/python src/analyze_round12.py
# round 13 (the delta rule of the authors' code, --delta_mode code, where lambda_hat_min <= 0 happens)
.venv/bin/python src/sweep_round13.py --workers 13
.venv/bin/python src/analyze_round13.py
.venv/bin/python src/analyze_followup.py                # results/summary_followup.md, figures/fig5-7
```
