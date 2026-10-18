"""Figure: volume only vs image check vs zero-shot VLM (reads results/main_plus_vlm_auroc.csv)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t = pd.read_csv("results/main_plus_vlm_auroc.csv")
KINDS = ["size", "shift", "island"]
CHECKS = [("volume", "volume only", "#7a8a99"), ("image_check", "image check", "#c2410c"),
          ("vlm", "VLM zero-shot (3B)", "#2f6f9f")]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
for ax, tgt in zip(axes, [0.9, 0.8]):
    for j, (c, lab, col) in enumerate(CHECKS):
        sub = t[(t.dice == tgt) & (t.check == c)].set_index("error").loc[KINDS]
        x = np.arange(len(KINDS)) + (j - 1) * 0.27
        ax.bar(x, sub.auroc, 0.25, color=col, label=lab,
               yerr=[sub.auroc - sub.ci_lo, sub.ci_hi - sub.auroc], capsize=2.5)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(KINDS)), ["wrong size", "shift", "extra island"])
    ax.set_title(f"Dice {tgt}")
    ax.set_ylim(0.3, 1.02)
axes[0].set_ylabel("AUROC (damaged vs clean)")
h, l = axes[0].get_legend_handles_labels()
fig.suptitle("Which damage does each check catch?")
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
fig.savefig("results/main_plus_vlm_figure.png", dpi=150)
print("saved results/main_plus_vlm_figure.png")
