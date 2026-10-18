# Design (simple version) — written before running

Facts marked **[measured]** were checked on the data. **[to verify]** must be confirmed by a pilot. Nothing here is a result yet.

## Question
Which segmentation-mask errors does a simple automatic check catch, and which does it miss?

## Method in one paragraph
Take masks that are (assumed) correct, damage them in three known ways at two known sizes, run two simple checks, and report how well each check separates damaged from undamaged masks (AUROC). The point is the list of blind spots, not a high score.

## Data
MSD Task04 Hippocampus, 260 cases, labels merged into one binary mask (CC-BY-SA 4.0, public). Method demo only; no clinical claim.
- Calibration of the checks: cases 0-99 (clean masks only; no labels are learned).
- Test: cases 100-259 (160 cases), each case gives one clean and one damaged version per (error, size). Evaluated once.
- Limit: all 260 cases were already used in the first round, where the "edge" check was changed to two-sided after looking at results. No other data exist (the other 130 MSD images have no labels), so a "fresh start" would use the same cases. To keep the test as clean as possible: the edge check is **fixed in advance as two-sided** (any abnormal edge strength is suspicious), the damage method, sizes and random seeds are new, and nothing is tuned on cases 100-259. The one-sided version appears only in a side table. Pilots use cases 0-59 only.
- All masks, clean or damaged, get the same light smoothing so the checks cannot use blocky-edge artefacts.

## The damage: three errors, size set by overlap with the clean mask (Dice)
| error | how | size levels |
|---|---|---|
| wrong size (too small or too big, random direction) | threshold of the smoothed mask moved up or down | Dice 0.9 and 0.8 **[measured]**: Dice hit exactly in 100% of 60 cases; volume change -18% / -33% (smaller), +22% / +50% (bigger) |
| shifted | whole mask moved by a sub-voxel amount in a random direction | Dice 0.9 and 0.8 **[to verify]** |
| extra island | a separate blob added away from the structure, volume set by Dice | Dice 0.9 and 0.8 **[to verify]**; at 0.8 the island is half the structure, which is extreme, said so in the report |
Why Dice: in the first round the sizes were in voxels, and the structure is only about 9 voxels thick at its thickest (median 8.7) **[measured]**, so "1 voxel" meant -41% volume for one error and almost nothing (Dice 0.99) for another.

## The two checks (score = how suspicious, higher = more suspicious)
1. **Volume only** (control, never sees the image): how far the mask volume is from normal. Population volume spread is 13% **[measured]**.
2. **Image check**: (a) how far the mask boundary's image edge strength is from normal (two-sided); (b) is the mask more than one connected piece. Score = the larger of the two.
Both are calibrated on cases 0-99 only. The two parts of the image check are also reported separately in a side table.

## Predictions, written before the test run (numbers fixed now; pilots P1 and P2 passed first)
- A1. Volume only, wrong size, Dice 0.8: AUROC >= 0.8.
- A2. Volume only, shift (both sizes): AUROC between 0.4 and 0.6 (shifting hardly changes volume; the pilot saw about -3%).
- A3. Image check, extra island (both sizes): AUROC >= 0.8.
- A4. Image check, shift, Dice 0.8: AUROC between 0.6 and 0.8.
- A5. Image check, wrong size, Dice 0.8: lower than the volume-only AUROC on the same item.
- A6. Shift at Dice 0.9: both checks below 0.7.
Each prediction is reported as confirmed / refuted by the point estimate, with its confidence interval.

## What is reported
One table: AUROC per error x size x check, with a 95% confidence interval from resampling cases (160 cases give a half-width of about 0.05-0.08). One figure of the same numbers. One paragraph: what each check misses. Conclusions are only drawn where the interval excludes 0.5.
One paragraph on the VLM from the first round, **no new runs**: local Qwen2.5-VL-3B, zero-shot, answered NO on 700/700 images, AUROC 0.27-0.50, answers changed with the wording (Spearman 0.23-0.57); the masks and prompts were not designed for it, so this says "not usable as set up", not "VLMs cannot do this".

## Steps
1. Pilot (go/no-go): shift and island hit the Dice targets within 0.01 for >= 95% of cases; look at 3 overlays per error and size to see that 0.9 is subtle and 0.8 is visible. If a pilot fails, fix the method and redo it; do not move on.
2. One full run on the test cases.
3. Table, figure, README, push to the private repo.

## Addendum (written before any VLM run on the new damage): the VLM as a third check
Why: a VLM check is the original point of the demo; the first-round VLM test used voxel-sized damage with very uneven difficulty and views that could not show a distant island, so it does not answer the question. The two-check main result above is unchanged and already final.
- **Same items as the main run**: test cases 100-259, one clean and six damaged versions (3 errors x Dice 0.9 / 0.8) per case, same damage seeds. Score = log P(NO) - log P(YES) at the first answer token, so no text parsing.
- **Model**: local Qwen2.5-VL-3B, zero-shot, one main prompt fixed now (the first-round prompt: "does the red outline correctly follow the boundary of one complete hippocampus? Answer YES or NO", plus one sentence describing the new fourth panel: "The fourth panel shows the mask alone, projected along one axis."). A second wording on a 40-case subset only, to measure prompt sensitivity. No other prompts will be tried.
- **Views**: three orthogonal slices through the centroid of the largest connected piece of the shown mask (what a reviewer would pick; no knowledge of where the damage is), plus one panel with the mask projected along one axis (so a distant island is visible).
- **Pilot first** (cases 0-59, 20 clean + 20 damaged per error x size): checks that the output format is stable (>= 95% of probability on YES/NO) and the speed. It does not decide whether the test run happens: the test run happens regardless, because "does the VLM work" needs a test answer. The pilot only decides whether a larger model is tried (see below).
- **Predictions (fixed now)**
  - V1. The 3B VLM is not useful: AUROC < 0.7 on all six (error, Dice) settings.
  - V2. Where it is best is the extra island at Dice 0.8 (the largest, most visible damage).
  - V3. Prompt agreement stays low: Spearman between the two wordings < 0.6.
  - V4. It does not beat the best of the two simple checks on any setting.
- **Decision rule, fixed now**: if the VLM reaches AUROC >= 0.8 with the CI lower bound above 0.5 on any setting, report it as useful for that error. If it stays < 0.7 everywhere, report "not usable zero-shot at 3B" and stop; a 7B model (about 5-16 GB download, needs the user's go-ahead) is tried only if the 3B shows some signal (CI lower bound > 0.5 on at least 2 settings).
- Cost: about 1120 images x ~2 s for the main prompt (about 40 minutes) plus about 280 for the second wording.
- Limits that stay: zero-shot only, no fine-tuning, one small dataset, synthetic damage.

## Not in this version (shelved in `archive/DESIGN_v2_shelved.md`)
Learned CNN inspector and held-out error types, prompt-sensitivity re-runs, the slab error, a locked third data split, the check on unmodified labels, a second dataset, a larger VLM.

## Limits
One small dataset; synthetic errors are not real annotator errors; the real-error size (Dice) is not known; two simple checks only; no clinical claim.
