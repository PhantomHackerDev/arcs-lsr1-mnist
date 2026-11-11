"""Pick each modern baseline's learning rate from the seed-0 sweep by final TRAINING loss
(the objective under study; the test set is not used for tuning).

  python src/select_lr.py modern_mnist 256   -> prints the chosen config names (comma-separated)
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prefix = {"modern_mnist": "deep", "modern_fmnist": "fm"}[sys.argv[1]]
bs = int(sys.argv[2])
best = {}
for f in glob.glob(os.path.join(ROOT, "results", f"{prefix}_*_lr*_bs{bs}_s0.json")):
    m = re.match(rf"({prefix}_(\w+?)_lr[\d.e-]+)_bs{bs}_s0\.json$", os.path.basename(f))
    if not m:
        continue
    cname, opt = m.group(1), m.group(2)
    with open(f) as fh:
        tl = json.load(fh)["epochs"][-1]["train_loss"]
    tl = tl if tl == tl else float("inf")  # NaN (diverged) loses
    if opt not in best or tl < best[opt][1]:
        best[opt] = (cname, tl)
for opt, (c, tl) in sorted(best.items()):
    print(f"{opt}: {c} (seed-0 train loss {tl:.4g})", file=sys.stderr)
print(",".join(c for c, _ in best.values()))
