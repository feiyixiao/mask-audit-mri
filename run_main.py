"""Main experiment (see DESIGN.md). Two simple checks vs. three damages at two Dice levels.

  python3 run_main.py --smoke   # code check on cases 0-39 (calibration cases, NOT a result)
  python3 run_main.py           # the real run: test cases 100-259, run once
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from audit.damage import clean_mask, damage
from audit.detectors import Calibrated
from audit.load_msd import load_cases

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

KINDS, TARGETS, B = ["size", "shift", "island"], [0.9, 0.8], 2000
cases, names = load_cases(seed=0)
ref = [(img, clean_mask(m)) for img, m in cases[:100]]
cal = Calibrated(ref)
idx = list(range(40)) if args.smoke else list(range(100, 260))
print(f"{'SMOKE (not a result)' if args.smoke else 'FINAL RUN'}: {len(idx)} cases, calibrated on cases 0-99")

CHECKS = {
    "volume": cal.blind,                 # never sees the image
    "image_check": cal.rules,            # max(two-sided edge, 3 x extra components)
    "edge_two_sided": cal.edge,          # parts of the image check (side table)
    "components": cal.components,
    "edge_one_sided": cal.edge_signed,   # first-round version (side table)
}


def score_all(img, m):
    return {k: f(img, m) for k, f in CHECKS.items()}


clean_scores = {}
dam_scores, vol = {}, {}
for k in idx:
    img, m = cases[k]
    cm = clean_mask(m)
    clean_scores[k] = score_all(img, cm)
    for ki, kind in enumerate(KINDS):
        for tgt in TARGETS:
            m2, d, _ = damage(m, kind, tgt, np.random.default_rng([k, ki, int(tgt * 100)]))
            dam_scores[(k, kind, tgt)] = score_all(img, m2)
            vol[(k, kind, tgt)] = (m2.sum() - cm.sum()) / cm.sum() * 100

rng = np.random.default_rng(0)
boot = [rng.integers(0, len(idx), len(idx)) for _ in range(B)]   # resample CASES


def auroc(kind, tgt, check, sel):
    cs = np.array([clean_scores[idx[i]][check] for i in sel])
    ds = np.array([dam_scores[(idx[i], kind, tgt)][check] for i in sel])
    return roc_auc_score(np.r_[np.zeros(len(cs)), np.ones(len(ds))], np.r_[cs, ds])


rows = []
full = np.arange(len(idx))
for kind in KINDS:
    for tgt in TARGETS:
        for check in CHECKS:
            a = auroc(kind, tgt, check, full)
            bs = [auroc(kind, tgt, check, s) for s in boot]
            rows.append(dict(error=kind, dice=tgt, check=check, auroc=a,
                             ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                             mean_volume_change_pct=np.mean([vol[(k, kind, tgt)] for k in idx])))
tab = pd.DataFrame(rows)
suffix = "_smoke" if args.smoke else ""
tab.to_csv(f"results/main_auroc{suffix}.csv", index=False)

show = tab.copy()
show["auroc (95% CI)"] = show.apply(lambda r: f"{r.auroc:.2f} ({r.ci_lo:.2f}-{r.ci_hi:.2f})", axis=1)
print(show.pivot_table(index=["error", "dice"], columns="check", values="auroc").round(2).to_string())

# ---- predictions A1-A6
def get(e, d, c):
    r = tab[(tab.error == e) & (tab.dice == d) & (tab.check == c)].iloc[0]
    return r.auroc, r.ci_lo, r.ci_hi


preds = [
    ("A1 volume, size, D0.8 >= 0.8", get("size", 0.8, "volume")[0] >= 0.8, get("size", 0.8, "volume")),
    ("A2 volume, shift D0.9 in [0.4,0.6]", 0.4 <= get("shift", 0.9, "volume")[0] <= 0.6, get("shift", 0.9, "volume")),
    ("A2 volume, shift D0.8 in [0.4,0.6]", 0.4 <= get("shift", 0.8, "volume")[0] <= 0.6, get("shift", 0.8, "volume")),
    ("A3 image, island D0.9 >= 0.8", get("island", 0.9, "image_check")[0] >= 0.8, get("island", 0.9, "image_check")),
    ("A3 image, island D0.8 >= 0.8", get("island", 0.8, "image_check")[0] >= 0.8, get("island", 0.8, "image_check")),
    ("A4 image, shift D0.8 in [0.6,0.8]", 0.6 <= get("shift", 0.8, "image_check")[0] <= 0.8, get("shift", 0.8, "image_check")),
    ("A5 image < volume, size D0.8", get("size", 0.8, "image_check")[0] < get("size", 0.8, "volume")[0],
     (get("size", 0.8, "image_check")[0], get("size", 0.8, "volume")[0], np.nan)),
    ("A6 shift D0.9 both < 0.7", max(get("shift", 0.9, "volume")[0], get("shift", 0.9, "image_check")[0]) < 0.7,
     (get("shift", 0.9, "volume")[0], get("shift", 0.9, "image_check")[0], np.nan)),
]
print("\nPredictions:")
for name, ok, v in preds:
    print(f"  {'CONFIRMED' if ok else 'REFUTED  '}  {name}   values {tuple(round(float(x), 2) for x in v)}")
pd.DataFrame([(n, o) for n, o, _ in preds], columns=["prediction", "confirmed"]).to_csv(
    f"results/main_predictions{suffix}.csv", index=False)

# ---- figure: the two main checks
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, tgt in zip(axes, TARGETS):
    for j, (check, col) in enumerate([("volume", "#7a8a99"), ("image_check", "#c2410c")]):
        sub = tab[(tab.dice == tgt) & (tab.check == check)].set_index("error").loc[KINDS]
        x = np.arange(len(KINDS)) + (j - 0.5) * 0.36
        ax.bar(x, sub.auroc, 0.34, color=col, label=check.replace("_", " "),
               yerr=[sub.auroc - sub.ci_lo, sub.ci_hi - sub.auroc], capsize=3)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(KINDS)), ["wrong size", "shift", "extra island"])
    ax.set_title(f"Dice {tgt}")
    ax.set_ylim(0.3, 1.02)
axes[0].set_ylabel("AUROC (damaged vs clean)")
axes[1].legend(loc="lower right", fontsize=8)
fig.suptitle(("SMOKE - not a result: " if args.smoke else "") + "Which damage does each check catch?")
fig.tight_layout()
fig.savefig(f"results/main_figure{suffix}.png", dpi=150)
print(f"saved results/main_auroc{suffix}.csv, main_predictions{suffix}.csv, main_figure{suffix}.png")
