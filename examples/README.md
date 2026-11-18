# GPU experiments: how to run them

Everything here runs with one command per step, writes its results under `results/cifar10/`, and can be stopped and resumed (finished runs are skipped).

## 1. Setup

```bash
git clone https://github.com/PhantomHackerDev/arcs-lsr1-mnist.git && cd arcs-lsr1-mnist
python3 -m venv .venv && . .venv/bin/activate
pip install torch torchvision          # the CUDA build for your driver, e.g. --index-url https://download.pytorch.org/whl/cu128
pip install numpy matplotlib
```

## 2. Pre-flight checks (a few minutes)

```bash
python src/test_math.py                          # expect "ALL PASS", including "[PASS] cuda == cpu (float64 state)"
python src/test_optim.py                         # torch.optim wrapper == functional implementation
python examples/test_examples.py --device cuda   # Muon baseline, ARCs-LSR1 on the GPU == on the CPU, every optimizer runs
```

The GPU checks could not be run on the development machine (no GPU). If any of them fails, please send the output.

## 3. Time one epoch (optional, to plan the sweep)

```bash
python examples/train_cifar10.py --device cuda --opt adamw --lr 1e-3 --epochs 1 --out /tmp/cifar_timing
python examples/train_cifar10.py --device cuda --opt arcs --history 50 --double_grad --epochs 1 --out /tmp/cifar_timing
```

The CIFAR-10 data (170 MB) is downloaded to `data/` on the first call. An ARCs-LSR1 epoch costs about two baseline epochs, because each step evaluates two gradients. The whole sweep is roughly 2,000 baseline-epoch equivalents, plus a full-training-set evaluation after every epoch. On one A100-class GPU that is very roughly 3–8 hours; the timing above gives a better estimate.

## 4. CIFAR-10 / ResNet-18 sweep

```bash
python examples/sweep_cifar10.py --gpus 0,1,2,3        # one job per GPU; --per_gpu 2 to share a GPU
python examples/analyze_cifar10.py                     # -> results/cifar10/summary.md, results/cifar10/fig_cifar10.png
```

The protocol is the same as rounds 9–12 on F-MNIST (see the main README). Batch 1024, no augmentation (`--augment` to turn it on), selection by the **training** loss on the full training set.

- **Baselines:** AdamW, SGD+momentum and Muon, each over 40 epochs with a learning-rate grid. The grid is extended by ×3 or ÷3 while the best value is at its edge. The learning rate is chosen once for epoch 20 (same epochs as ARCs-LSR1) and once for epoch 40 (same compute).
  - Muon: the same update as on F-MNIST, applied to all conv and linear weights except the classifier, with Adam for the rest.
- **ARCs-LSR1:** 20 epochs.
  - Settings: δ rule of the authors' code, model gradient over two batches, float32 L-SR1 buffers.
  - Grid: m ∈ {10, 50} × μ_min ∈ {0.25, 1, 4}, plus the diagonal B₀ (m = 50, μ_min ∈ {0.25, 1}). A best μ_min at the edge is extended.
- **Seeds:** 1 and 2 at every selected configuration.

Memory: the L-SR1 buffer is 2m × 11.2M values, i.e. 4.5 GB at m = 50 in float32, in addition to the model and activations.

## 5. Sending the results back

```bash
git checkout -b cifar10-results
git add results/cifar10/*.json results/cifar10/summary.md results/cifar10/fig_cifar10.png
git commit -m "CIFAR-10 results" && git push -u origin cifar10-results
```

The per-run logs in `results/cifar10/logs/` are not committed (they are in `.gitignore`). If a run fails, its log there shows why.

## Files

| file | what it does |
|---|---|
| `train_cifar10.py` | one run: ResNet-18 on CIFAR-10 with ARCs-LSR1 (`ARCsLSR1Optim`), AdamW, SGD or Muon; data kept on the GPU; JSON log per run |
| `sweep_cifar10.py` | the sweep above: grid → extension of edge values → selection → seeds, spread over GPUs; `--dry_run` lists the jobs, `--smoke --gpus cpu` runs the whole pipeline on random data |
| `analyze_cifar10.py` | tables (per epoch and at equal compute) and the figure |
| `optim_baselines.py` | Muon for `nn.Module`s (conv weights included), identical to the flat version used on F-MNIST |
| `test_examples.py` | the pre-flight checks |
