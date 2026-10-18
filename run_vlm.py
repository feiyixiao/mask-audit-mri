"""VLM inspector vs. the other inspectors, on the same (sub-sampled) items.

  python3 run_vlm.py --smoke      # 4 clean + 2 per corruption, prints timing
  python3 run_vlm.py              # 100 clean + 40 per (corruption, severity 2 and 3)

Per-item VLM scores are cached in results/vlm_items.csv, so a re-run only computes what is missing.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from audit.corrupt import CORRUPTIONS
from audit.detectors import Calibrated
from audit.load_msd import load_cases

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--n_clean", type=int, default=100)
ap.add_argument("--n_bad", type=int, default=40)
ap.add_argument("--severities", type=int, nargs="+", default=[2, 3])
args = ap.parse_args()
if args.smoke:
    args.n_clean, args.n_bad, args.severities = 4, 2, [3]

cases, names = load_cases(seed=0)
ref, test = cases[:100], cases[100:]
test_names = names[100:]
cal = Calibrated(ref)

# ---- build the item list: (case index in test, corruption, severity)
items = [(i, "clean", 0) for i in range(args.n_clean)]
for ctype in CORRUPTIONS:
    for sev in args.severities:
        items += [(args.n_clean + i, ctype, sev) for i in range(args.n_bad)]


def build_mask(i, ctype, sev):
    img, m = test[i]
    if ctype == "clean":
        return img, m
    rng = np.random.default_rng([i, list(CORRUPTIONS).index(ctype), sev])
    return img, CORRUPTIONS[ctype](m, sev, rng)


cache_path = Path("results/vlm_items.csv")
cache = pd.read_csv(cache_path) if cache_path.exists() and not args.smoke else pd.DataFrame()
done = set(zip(cache.case, cache.corruption, cache.severity)) if len(cache) else set()
todo = [(i, c, s) for i, c, s in items if (test_names[i], c, s) not in done]
print(f"{len(items)} items, {len(todo)} still to score with the VLM")

rows = []
if todo:
    from audit.vlm import VLMInspector
    t0 = time.time()
    vlm = VLMInspector()
    print(f"model loaded in {time.time() - t0:.1f}s on {vlm.device}")
    for n, (i, c, s) in enumerate(todo, 1):
        img, m = build_mask(i, c, s)
        t1 = time.time()
        score, mass = vlm.score_image(__import__("audit.overlay", fromlist=["render"]).render(img, m))
        rows.append(dict(case=test_names[i], corruption=c, severity=s, vlm=score, yes_no_mass=mass))
        if n <= 5 or n % 25 == 0:
            print(f"  [{n}/{len(todo)}] {c} s{s}: score {score:+.2f}, P(yes|no) mass {mass:.2f}, {time.time() - t1:.1f}s")
        if n % 25 == 0 and not args.smoke:
            pd.concat([cache, pd.DataFrame(rows)]).to_csv(cache_path, index=False)
    new = pd.DataFrame(rows)
    cache = pd.concat([cache, new]) if len(cache) else new
    if not args.smoke:
        cache.to_csv(cache_path, index=False)

# ---- add the other inspectors on exactly the same items, then AUROC vs. the clean items
res = []
for i, c, s in items:
    img, m = build_mask(i, c, s)
    r = dict(case=test_names[i], corruption=c, severity=s,
             blind=cal.blind(img, m), edge=cal.edge(img, m), rules=cal.rules(img, m),
             combined=cal.combined(img, m))
    res.append(r)
res = pd.DataFrame(res).merge(cache[["case", "corruption", "severity", "vlm", "yes_no_mass"]],
                              on=["case", "corruption", "severity"], how="left")
clean = res[res.corruption == "clean"]
print(f"\nVLM answered YES/NO with mean probability mass {res.yes_no_mass.mean():.2f} "
      "(low = it wants to say something else)")
print(f"clean items: VLM score mean {clean.vlm.mean():+.2f}, sd {clean.vlm.std():.2f}\n")

out = []
for (c, s), g in res[res.corruption != "clean"].groupby(["corruption", "severity"]):
    y = np.r_[np.zeros(len(clean)), np.ones(len(g))]
    row = dict(corruption=c, severity=s)
    for d in ["vlm", "blind", "edge", "rules", "combined"]:
        row[d] = roc_auc_score(y, np.r_[clean[d], g[d]])
    out.append(row)
tab = pd.DataFrame(out)
print(tab.round(2).to_string(index=False))
if not args.smoke:
    tab.to_csv("results/vlm_vs_baselines_auroc.csv", index=False)
    res.to_csv("results/vlm_items_with_baselines.csv", index=False)
