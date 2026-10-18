"""End-to-end check of the audit pipeline on synthetic data (no downloads).
Run:  python3 run_synthetic.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audit.synth import make_case
from audit.evaluate import run

rng = np.random.default_rng(42)
cases = [make_case(rng) for _ in range(160)]
df = run(cases, n_ref=60, severities=(1, 2, 3), seed=1)

df.to_csv("results/synthetic_auroc.csv", index=False)
table = df.pivot_table(index=["corruption", "severity"], columns="detector", values="auroc")
print(table.round(2).to_string())

# heatmap: detector x (corruption, severity)
order = ["blind", "edge", "components", "rules", "combined"]
t = table[order]
fig, ax = plt.subplots(figsize=(6.5, 7))
im = ax.imshow(t.values, vmin=0.4, vmax=1.0, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(order)), order, rotation=30)
ax.set_yticks(range(len(t)), [f"{c} s{s}" for c, s in t.index])
for r in range(t.shape[0]):
    for c in range(t.shape[1]):
        ax.text(c, r, f"{t.values[r, c]:.2f}", ha="center", va="center", color="w", fontsize=8)
fig.colorbar(im, label="AUROC (corrupted vs clean)")
ax.set_title("Synthetic sanity check: who catches which mask error?")
fig.tight_layout()
fig.savefig("results/synthetic_heatmap.png", dpi=150)
print("saved results/synthetic_heatmap.png")
