"""Learned inspector (3D CNN) vs. rule baselines vs. zero-shot VLM.

  python3 run_learned.py --quick     # tiny run to check the code (not a result)
  python3 run_learned.py             # 3 seeds; models: all types, each type held out, shuffled-label control

Splits (by case): train = cases[:80], val = cases[80:100], test = cases[100:] (same test cases as
the baselines). Test items are the same as in run_vlm.py: 100 clean vs 60 corrupted per (type, severity).
All masks, clean or corrupted, are smoothed the same way before any inspector sees them.
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from audit.detectors import Calibrated
from audit.learned import TYPES, make_input, pick_device, predict, smooth_mask, train, corrupt
from audit.load_msd import load_cases

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--seeds", type=int, default=3)
args = ap.parse_args()
EPOCHS, STEPS = (2, 2) if args.quick else (40, 12)
SEEDS = [0] if args.quick else list(range(args.seeds))
SEVS = [2, 3]
N_CLEAN, N_BAD = (10, 6) if args.quick else (100, 60)

cases, names = load_cases(seed=0)
train_cases, val_cases, test = cases[:80], cases[80:100], cases[100:]
dev = pick_device()
print(f"device {dev} | train {len(train_cases)} val {len(val_cases)} test {len(test)} cases")

# ---- test items (identical corruption seeds as run_vlm.py), smoothed like everything else
items = [(i, "clean", 0) for i in range(N_CLEAN)]
for c in TYPES:
    for s in SEVS:
        items += [(N_CLEAN + i, c, s) for i in range(N_BAD)]
pairs, inputs = [], []
for i, c, s in items:
    img, m = test[i]
    if c != "clean":
        m = corrupt(m, c, s, np.random.default_rng([i, TYPES.index(c), s]))
    m = smooth_mask(m)
    pairs.append((img, m)); inputs.append(make_input(img, m))
meta = pd.DataFrame(items, columns=["i", "corruption", "severity"])

# ---- rule baselines on the same (smoothed) masks, calibrated on smoothed reference cases
cal = Calibrated([(img, smooth_mask(m)) for img, m in cases[:100]])
for d in ["blind", "edge", "rules"]:
    meta[d] = [getattr(cal, d)(img, m) for img, m in pairs]

# ---- learned inspectors
runs = {"all": TYPES, "shuffled_labels": TYPES}
for t in TYPES:
    runs[f"without_{t}"] = [x for x in TYPES if x != t]
val_auc = []
for name, ttypes in runs.items():
    for seed in SEEDS:
        t0 = time.time()
        model, best = train(train_cases, val_cases, ttypes, seed, epochs=EPOCHS, steps=STEPS,
                            shuffle_labels=(name == "shuffled_labels"), device=dev)
        meta[f"{name}__{seed}"] = predict(model, inputs, dev)
        val_auc.append(dict(model=name, seed=seed, best_val_auroc=best))
        print(f"  {name:22s} seed {seed}: best val AUROC {best:.3f} ({time.time() - t0:.0f}s)", flush=True)
pd.DataFrame(val_auc).to_csv("results/learned_val_auroc.csv", index=False)
meta.to_csv("results/learned_item_scores.csv", index=False)

# ---- AUROC per (corruption, severity)
clean = meta[meta.corruption == "clean"]
rows = []
for (c, s), g in meta[meta.corruption != "clean"].groupby(["corruption", "severity"]):
    y = np.r_[np.zeros(len(clean)), np.ones(len(g))]

    def auc(col):
        return roc_auc_score(y, np.r_[clean[col], g[col]])

    def seeds_mean(prefix):
        v = [auc(f"{prefix}__{k}") for k in SEEDS]
        return np.mean(v), np.std(v)

    row = dict(corruption=c, severity=int(s), blind=auc("blind"), edge=auc("edge"), rules=auc("rules"))
    row["learned_seen"], row["learned_seen_sd"] = seeds_mean("all")
    row["learned_unseen"], row["learned_unseen_sd"] = seeds_mean(f"without_{c}")
    row["shuffled_control"], _ = seeds_mean("shuffled_labels")
    rows.append(row)
tab = pd.DataFrame(rows)
try:   # zero-shot VLM numbers from run_vlm.py (computed on the unsmoothed masks)
    v = pd.read_csv("results/vlm_vs_baselines_auroc.csv")[["corruption", "severity", "vlm"]]
    tab = tab.merge(v, on=["corruption", "severity"], how="left")
except FileNotFoundError:
    pass
print()
print(tab.round(2).to_string(index=False))
if not args.quick:
    tab.to_csv("results/learned_vs_baselines_auroc.csv", index=False)
