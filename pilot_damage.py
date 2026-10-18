"""Pilot P1: do the three damage methods hit the Dice targets? (cases 0-59 only; test cases untouched)
Also saves example overlays so we can look at how visible each error and size is.
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from audit.damage import clean_mask, damage
from audit.load_msd import load_cases
from audit.overlay import render

cases, names = load_cases(seed=0)
KINDS, TARGETS = ["size", "shift", "island"], [0.9, 0.8]
rows, shown = [], {}
for k, (img, m) in enumerate(cases[:60]):
    clean = clean_mask(m)
    for ki, kind in enumerate(KINDS):
        for tgt in TARGETS:
            rng = np.random.default_rng([k, ki, int(tgt * 100)])
            m2, d, info = damage(m, kind, tgt, rng)
            rows.append(dict(case=k, kind=kind, target=tgt, achieved=d, hit=abs(d - tgt) <= 0.01,
                             vol_change_pct=(m2.sum() - clean.sum()) / clean.sum() * 100))
            if k < 3:
                shown[(kind, tgt, k)] = (img, m2)
df = pd.DataFrame(rows)
df.to_csv("results/pilot_damage.csv", index=False)
print(df.groupby(["kind", "target"]).agg(
    hit_rate=("hit", "mean"), dice_mean=("achieved", "mean"), dice_min=("achieved", "min"),
    dice_max=("achieved", "max"), vol_change_pct=("vol_change_pct", "mean")).round(3).to_string())
print("\nfailed cases (|achieved - target| > 0.01):")
print(df[~df.hit].groupby(["kind", "target"]).size().to_string() or "none")

# overlay sheet: clean + (kind x target) for case 0, 1, 2 (rows = case, error, size)
tiles = []
for k in range(3):
    img, m = cases[k]
    tiles.append((f"case{k} clean", render(img, clean_mask(m))))
    for kind in KINDS:
        for tgt in TARGETS:
            tiles.append((f"case{k} {kind} D{tgt}", render(*shown[(kind, tgt, k)])))
W, H = tiles[0][1].size
sheet = Image.new("RGB", (W + 130, H * len(tiles)), (30, 30, 30))
dr = ImageDraw.Draw(sheet)
for i, (lab, im) in enumerate(tiles):
    sheet.paste(im, (130, i * H)); dr.text((6, i * H + H // 2), lab, fill=(255, 255, 255))
sheet.save("results/pilot_overlays.png")
print("\nsaved results/pilot_overlays.png", sheet.size)
