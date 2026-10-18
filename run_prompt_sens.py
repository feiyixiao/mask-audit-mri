"""Prompt sensitivity of the VLM inspector: same images, three different questions.
If the AUROC or the item ranking changes a lot with wording, the VLM result is not trustworthy.

  python3 run_prompt_sens.py
Output: results/prompt_sensitivity_auroc.csv, results/prompt_sensitivity_corr.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from audit.corrupt import CORRUPTIONS
from audit.load_msd import load_cases
from audit.overlay import render
from audit.vlm import PROMPT, VLMInspector

PROMPTS = {
    "A_original": (PROMPT, False),
    "B_qc_wording": (
        "This is a quality-control check of a brain MRI segmentation. Three slices are shown, "
        "with the hippocampus segmentation outlined in red. Is the segmentation accurate? "
        "Answer with exactly one word: YES or NO.", False),
    "C_error_framing": (
        "The red contour is meant to outline the hippocampus on these three MRI slices. "
        "Is there any error in the contour (too large, too small, shifted, a missing part, "
        "or an extra region)? Answer with exactly one word: YES or NO.", True),  # YES = error
}
N_CLEAN, N_BAD, SEV = 60, 30, 3

cases, names = load_cases(seed=0)
test = cases[100:]
items = [(i, "clean") for i in range(N_CLEAN)]
for c in CORRUPTIONS:
    items += [(N_CLEAN + i, c) for i in range(N_BAD)]

vlm = VLMInspector()
rows = []
for i, c in items:
    img, m = test[i]
    if c != "clean":
        m = CORRUPTIONS[c](m, SEV, np.random.default_rng([i, list(CORRUPTIONS).index(c), SEV]))
    pic = render(img, m)
    for pname, (prompt, flip) in PROMPTS.items():
        vlm.prompt = prompt
        s, _ = vlm.score_image(pic)
        rows.append(dict(i=i, corruption=c, prompt=pname, score=-s if flip else s))
df = pd.DataFrame(rows)
df.to_csv("results/prompt_sensitivity_items.csv", index=False)

out = []
for pname, g in df.groupby("prompt"):
    clean = g[g.corruption == "clean"].score
    for c in CORRUPTIONS:
        bad = g[g.corruption == c].score
        y = np.r_[np.zeros(len(clean)), np.ones(len(bad))]
        out.append(dict(prompt=pname, corruption=c, auroc=roc_auc_score(y, np.r_[clean, bad])))
tab = pd.DataFrame(out).pivot(index="corruption", columns="prompt", values="auroc")
print(tab.round(2).to_string())
tab.to_csv("results/prompt_sensitivity_auroc.csv")

wide = df.pivot_table(index=["i", "corruption"], columns="prompt", values="score")
corr = wide.corr(method="spearman")
print("\nSpearman correlation of item scores between prompts:")
print(corr.round(2).to_string())
corr.to_csv("results/prompt_sensitivity_corr.csv")
