"""Step 0 diagnostics: why is the zero-shot VLM at chance? (DESIGN.md, 'Step 0 diagnostics')
Cases 0-29 only; the test cases are not touched.
  python3 diag_vlm_step0.py
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from audit.damage import clean_mask, damage, dice
from audit.load_msd import load_cases
from audit.overlay import PANEL, render_v2
from audit.vlm import PROMPT, VLMInspector

N, B = 30, 1000
KINDS, LADDER = ["size", "shift", "island"], [0.9, 0.8, 0.6, 0.4]
MAIN = PROMPT + " The fourth panel shows the mask alone, projected along one axis."
COUNT = ("The picture shows a white shape on a black background. How many separate white shapes "
         "are there? Answer with a single digit.")
LINE = "Is there a red line in this image? Answer with exactly one word: YES or NO."
YES, NO = ["YES", "Yes", "yes"], ["NO", "No", "no"]

cases, _ = load_cases(seed=0)
vlm = VLMInspector()
print("model loaded", vlm.device, flush=True)
rows = []
t0 = time.time()
n_calls = 0


def rec(test, setting, case, score, mass, **extra):
    global n_calls
    n_calls += 1
    rows.append(dict(test=test, setting=setting, case=case, score=score, mass=mass, **extra))
    if n_calls % 50 == 0:
        print(f"  {n_calls} calls, {(time.time() - t0) / 60:.1f} min", flush=True)


def panel_crop(im, i):
    return im.crop((i * (PANEL + 4), 0, i * (PANEL + 4) + PANEL, PANEL))


masks = {}   # (case, setting) -> (mask, achieved dice)
for k in range(N):
    img, m = cases[k]
    cm = clean_mask(m)
    masks[(k, "clean")] = (cm, 1.0)
    for ki, kind in enumerate(KINDS):
        for tgt in LADDER:
            m2, d, _ = damage(m, kind, tgt, np.random.default_rng([k, ki, int(tgt * 100)]))
            masks[(k, f"{kind}_{tgt}")] = (m2, d)
    other = clean_mask(cases[(k + 50) % 100][1])
    wm = np.zeros_like(cm)
    s = tuple(min(a, b) for a, b in zip(wm.shape, other.shape))
    wm[:s[0], :s[1], :s[2]] = other[:s[0], :s[1], :s[2]]
    masks[(k, "wrong_mask")] = (wm, dice(cm, wm))

# ---------------- D3b: can it see a thin red line at all?
for k in range(N):
    img, _ = cases[k]
    cm = masks[(k, "clean")][0]
    for name, outline in [("with_outline", True), ("without_outline", False)]:
        pic = panel_crop(render_v2(img, cm, outline=outline), 0)
        s, mass = vlm.score_tokens(pic, LINE, YES, NO)
        rec("D3b_line", name, k, s, mass)
# ---------------- D3a: counting separate shapes in the projection panel
for k in range(N):
    img, _ = cases[k]
    for setting in ["clean"] + [f"island_{t}" for t in (0.9, 0.8, 0.6)]:
        mk, d = masks[(k, setting)]
        pic = panel_crop(render_v2(img, mk), 3)
        s, mass = vlm.score_tokens(pic, COUNT, ["2"], ["1"])
        rec("D3a_count", setting, k, s, mass, dice=d)
# ---------------- D2: does the answer depend on the outline?
for k in range(N):
    img, _ = cases[k]
    pic = render_v2(img, masks[(k, "clean")][0], outline=False)
    s, mass = vlm.score_tokens(pic, MAIN, NO, YES)
    rec("D2_outline", "no_outline", k, s, mass)
# ---------------- D1: difficulty ladder (clean with outline is the reference)
for k in range(N):
    img, _ = cases[k]
    for setting in ["clean", "wrong_mask"] + [f"{kd}_{t}" for kd in KINDS for t in LADDER]:
        mk, d = masks[(k, setting)]
        s, mass = vlm.score_tokens(render_v2(img, mk), MAIN, NO, YES)
        rec("D1_ladder", setting, k, s, mass, dice=d)

df = pd.DataFrame(rows)
df.to_csv("results/diag_step0_items.csv", index=False)

# ---------------- analysis
rng = np.random.default_rng(0)
boot = [rng.integers(0, N, N) for _ in range(B)]


def get(test, setting):
    return df[(df.test == test) & (df.setting == setting)].sort_values("case").score.values


def auc(neg, pos, sel=slice(None)):
    n_, p_ = np.asarray(neg)[sel], np.asarray(pos)[sel]
    return roc_auc_score(np.r_[np.zeros(len(n_)), np.ones(len(p_))], np.r_[n_, p_])


def ci(neg, pos):
    b = [auc(neg, pos, s) for s in boot]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


out = []


def add(test, setting, neg, pos, note=""):
    a = auc(neg, pos); lo, hi = ci(neg, pos)
    out.append(dict(test=test, setting=setting, auroc=a, ci_lo=lo, ci_hi=hi, note=note))


clean_main = get("D1_ladder", "clean")
for setting in ["wrong_mask"] + [f"{kd}_{t}" for kd in KINDS for t in LADDER]:
    d = df[(df.test == "D1_ladder") & (df.setting == setting)].dice.mean()
    add("D1_ladder", setting, clean_main, get("D1_ladder", setting), f"mean achieved Dice {d:.2f}")
add("D2_outline", "no_outline vs outline", clean_main, get("D2_outline", "no_outline"),
    "score = log P(NO) - log P(YES); same slices without red line")
for t in (0.9, 0.8, 0.6):
    add("D3a_count", f"island_{t}", get("D3a_count", "clean"), get("D3a_count", f"island_{t}"),
        "score = log P('2') - log P('1')")
add("D3b_line", "with vs without outline", get("D3b_line", "without_outline"), get("D3b_line", "with_outline"),
    "score = log P(YES) - log P(NO)")
res = pd.DataFrame(out)
res.to_csv("results/diag_step0_auroc.csv", index=False)
pd.set_option("display.width", 200)
print("\nAUROC (n = %d cases per side):" % N)
print(res.round(2).to_string(index=False))

cnt_acc_clean = (get("D3a_count", "clean") < 0).mean()
cnt_acc_isl = (get("D3a_count", "island_0.8") > 0).mean()
print(f"\ncounting accuracy: clean says '1' {cnt_acc_clean:.2f}, island(0.8) says '2' {cnt_acc_isl:.2f}")
print(f"answer-format mass (mean over all calls): {df.mass.mean():.3f}")

g = lambda t, s: res[(res.test == t) & (res.setting == s)].auroc.iloc[0]
wrong = g("D1_ladder", "wrong_mask")
d2 = g("D2_outline", "no_outline vs outline"); d2 = max(d2, 1 - d2)
small = max(g("D1_ladder", f"{kd}_{t}") for kd in KINDS for t in (0.9, 0.8))
print("\nPredictions:")
for name, ok, v in [("P1 wrong-mask AUROC < 0.8", wrong < 0.8, wrong),
                    ("P2 counting AUROC >= 0.8 at island Dice 0.8", g("D3a_count", "island_0.8") >= 0.8, g("D3a_count", "island_0.8")),
                    ("P3 outline vs no outline, distinguishability < 0.6", d2 < 0.6, d2),
                    ("P4 sees the red line, AUROC >= 0.9", g("D3b_line", "with vs without outline") >= 0.9, g("D3b_line", "with vs without outline"))]:
    print(f"  {'CONFIRMED' if ok else 'REFUTED  '}  {name}   value {v:.2f}")
if wrong < 0.65 and d2 < 0.6:
    branch = "(a) does not use the outline for this question -> recognise-what-is-inside use or a trained probe"
elif wrong >= 0.8 and small < 0.7:
    branch = "(b) sees gross errors only -> targeted questions and a few worked examples, then a probe"
else:
    branch = "(c) unclear -> rerun D1 with more cases"
print(f"\nBranch by the pre-set rule: {branch}   (wrong-mask {wrong:.2f}, outline distinguishability {d2:.2f}, best Dice 0.9/0.8 {small:.2f})")
print(f"done in {(time.time() - t0) / 60:.1f} min")
