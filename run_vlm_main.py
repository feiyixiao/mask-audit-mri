"""VLM as the third check (see DESIGN.md, Addendum). Same test cases and damage as run_main.py.

  python3 run_vlm_main.py --pilot   # cases 0-59: format and speed only
  python3 run_vlm_main.py           # test cases 100-259 (+ second wording on 40 of them); resumable
Scores are cached in results/vlm2_items.csv.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from audit.damage import clean_mask, damage
from audit.load_msd import load_cases
from audit.overlay import render_v2
from audit.vlm import PROMPT, VLMInspector

ap = argparse.ArgumentParser()
ap.add_argument("--pilot", action="store_true")
args = ap.parse_args()

PROMPTS = {
    "main": PROMPT + " The fourth panel shows the mask alone, projected along one axis.",
    "alt": ("This is a quality-control check of a brain MRI segmentation. The first three panels are "
            "MRI slices with the hippocampus segmentation outlined in red; the fourth panel shows the "
            "mask alone, projected along one axis. Is the segmentation accurate? "
            "Answer with exactly one word: YES or NO."),
}
KINDS, TARGETS, B = ["size", "shift", "island"], [0.9, 0.8], 2000
N_ALT = 40
cases, names = load_cases(seed=0)

# ---------------------------------------------------------------- build the item list
if args.pilot:
    items = [(k, "clean", 0.0) for k in range(20)]
    for kind in KINDS:
        for tgt in TARGETS:
            items += [(k, kind, tgt) for k in range(20, 40)]
    idx_for_alt = []
else:
    idx = list(range(100, 260))
    items = [(k, "clean", 0.0) for k in idx]
    for k in idx:
        items += [(k, kind, tgt) for kind in KINDS for tgt in TARGETS]
    idx_for_alt = idx[:N_ALT]


def build(k, kind, tgt):
    img, m = cases[k]
    if kind == "clean":
        return img, clean_mask(m)
    m2, _, _ = damage(m, kind, tgt, np.random.default_rng([k, KINDS.index(kind), int(tgt * 100)]))
    return img, m2


cache_path = Path("results/vlm2_pilot_items.csv" if args.pilot else "results/vlm2_items.csv")
cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(
    columns=["case", "kind", "dice", "prompt", "score", "mass"])
done = set(zip(cache.case, cache.kind, cache.dice, cache.prompt))
todo = [(k, kind, tgt, "main") for k, kind, tgt in items if (k, kind, tgt, "main") not in done]
todo += [(k, kind, tgt, "alt") for k, kind, tgt in items if k in idx_for_alt and (k, kind, tgt, "alt") not in done]
print(f"{len(items)} items; {len(todo)} VLM calls still to do", flush=True)

rows, t0, vlm = [], time.time(), None
if todo:
    vlm = VLMInspector()
    print(f"model loaded on {vlm.device}", flush=True)
for n, (k, kind, tgt, pname) in enumerate(todo, 1):
    vlm.prompt = PROMPTS[pname]
    s, mass = vlm.score_image(render_v2(*build(k, kind, tgt)))
    rows.append(dict(case=k, kind=kind, dice=tgt, prompt=pname, score=s, mass=mass))
    if n % 50 == 0 or n == len(todo):
        el = time.time() - t0
        print(f"  [{n}/{len(todo)}] {el / 60:.1f} min elapsed, {el / n:.2f} s/call", flush=True)
        pd.concat([cache, pd.DataFrame(rows)]).to_csv(cache_path, index=False)
cache = pd.concat([cache, pd.DataFrame(rows)]) if rows else cache
if rows:
    cache.to_csv(cache_path, index=False)

main = cache[cache.prompt == "main"]
main = main[main.case.isin([k for k, _, _ in items])]
print(f"\nanswer format: YES/NO probability mass mean {main.mass.mean():.3f}, min {main.mass.min():.3f}, "
      f"share of items with mass >= 0.95: {(main.mass >= 0.95).mean():.3f}")
print(f"share of items where P(NO) > P(YES): {(main.score > 0).mean():.3f}")

# ---------------------------------------------------------------- AUROC per (kind, dice)
clean_sc = main[main.kind == "clean"].set_index("case").score
case_list = sorted(clean_sc.index)
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(case_list), len(case_list)) for _ in range(B)]


def auroc(kind, tgt, sel):
    if args.pilot:   # pilot: clean and damaged items come from different cases, so unpaired
        d_ = main[(main.kind == kind) & (main.dice == tgt)].score.values
        c_ = main[main.kind == "clean"].score.values
        return roc_auc_score(np.r_[np.zeros(len(c_)), np.ones(len(d_))], np.r_[c_, d_])
    cs = np.array([clean_sc[case_list[i]] for i in sel])
    dm = main[(main.kind == kind) & (main.dice == tgt)].set_index("case").score
    ds = np.array([dm[case_list[i]] for i in sel])
    return roc_auc_score(np.r_[np.zeros(len(cs)), np.ones(len(ds))], np.r_[cs, ds])


rows = []
for kind in KINDS:
    for tgt in TARGETS:
        a = auroc(kind, tgt, np.arange(len(case_list)))
        bs = [auroc(kind, tgt, s) for s in boot] if not args.pilot else [np.nan]
        rows.append(dict(error=kind, dice=tgt, check="vlm", auroc=a,
                         ci_lo=np.nanpercentile(bs, 2.5), ci_hi=np.nanpercentile(bs, 97.5)))
vt = pd.DataFrame(rows)
print("\nVLM AUROC (n = %d cases%s):" % (len(case_list), ", PILOT - not a result" if args.pilot else ""))
print(vt.round(2).to_string(index=False))
if args.pilot:
    raise SystemExit

vt.to_csv("results/vlm2_auroc.csv", index=False)
base = pd.read_csv("results/main_auroc.csv")
base = base[base.check.isin(["volume", "image_check"])][["error", "dice", "check", "auroc", "ci_lo", "ci_hi"]]
allt = pd.concat([base, vt])
allt.to_csv("results/main_plus_vlm_auroc.csv", index=False)

# ---------------------------------------------------------------- predictions V1-V4
def g(e, d, c):
    r = allt[(allt.error == e) & (allt.dice == d) & (allt.check == c)].iloc[0]
    return r.auroc


vlm_vals = {(e, d): g(e, d, "vlm") for e in KINDS for d in TARGETS}
alt = cache[cache.prompt == "alt"]
mm = main.merge(alt, on=["case", "kind", "dice"], suffixes=("_main", "_alt"))
rho = spearmanr(mm.score_main, mm.score_alt)[0] if len(mm) > 2 else np.nan
best_simple = {(e, d): max(g(e, d, "volume"), g(e, d, "image_check")) for e in KINDS for d in TARGETS}
preds = [
    ("V1 VLM < 0.7 on all six settings", all(v < 0.7 for v in vlm_vals.values()), max(vlm_vals.values())),
    ("V2 best VLM setting is island at Dice 0.8", max(vlm_vals, key=vlm_vals.get) == ("island", 0.8),
     vlm_vals[("island", 0.8)]),
    ("V3 prompt agreement Spearman < 0.6", rho < 0.6, rho),
    ("V4 VLM never beats the best simple check", all(vlm_vals[k] <= best_simple[k] for k in vlm_vals),
     max(vlm_vals[k] - best_simple[k] for k in vlm_vals)),
]
print("\nPredictions:")
for n_, ok, v in preds:
    print(f"  {'CONFIRMED' if ok else 'REFUTED  '}  {n_}   value {v:.2f}")
pd.DataFrame([(n_, o) for n_, o, _ in preds], columns=["prediction", "confirmed"]).to_csv(
    "results/vlm2_predictions.csv", index=False)
print("\nfinal table (AUROC):")
print(allt.pivot_table(index=["error", "dice"], columns="check", values="auroc").round(2).to_string())
