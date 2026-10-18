"""Audit pipeline on real MRI + hippocampus masks (MSD Task04, CC-BY-SA 4.0).
Run:  python3 run_msd.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audit.load_msd import load_cases
from audit.evaluate import run

import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # which shuffle -> which cases are reference/test
cases, names = load_cases(seed=SEED)
shapes = np.array([c[0].shape for c in cases])
vols = np.array([c[1].sum() for c in cases])
print(f"{len(cases)} cases | shape min {shapes.min(0)} max {shapes.max(0)} | "
      f"mask volume voxels: median {np.median(vols):.0f}, range {vols.min()}-{vols.max()}")

df = run(cases, n_ref=100, severities=(1, 2, 3), seed=1)
df.to_csv(f"results/msd_auroc_seed{SEED}.csv", index=False)
table = df.pivot_table(index=["corruption", "severity"], columns="detector", values="auroc")
order = ["blind", "edge_signed", "edge", "components", "rules", "combined"]
print(table[order].round(2).to_string())

t = table[order]
fig, ax = plt.subplots(figsize=(6.5, 7))
im = ax.imshow(t.values, vmin=0.4, vmax=1.0, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(order)), order, rotation=30)
ax.set_yticks(range(len(t)), [f"{c} s{s}" for c, s in t.index])
for r in range(t.shape[0]):
    for c in range(t.shape[1]):
        ax.text(c, r, f"{t.values[r, c]:.2f}", ha="center", va="center", color="w", fontsize=8)
fig.colorbar(im, label="AUROC (corrupted vs clean)")
ax.set_title("MSD Hippocampus: who catches which mask error?")
fig.tight_layout()
fig.savefig(f"results/msd_heatmap_seed{SEED}.png", dpi=150)
print(f"saved results/msd_heatmap_seed{SEED}.png")
