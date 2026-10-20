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

## Step 0 diagnostics (after the VLM was at chance; written before running; cases 0-29 only, test cases untouched)
Question: *why* is the zero-shot VLM at chance? Four small tests, same model, same four-panel view, same main prompt unless stated. 30 cases each; AUROC with a case-bootstrap CI.
- **D1 difficulty ladder.** Each error at Dice 0.9, 0.8, 0.6, 0.4 (as far as the method can reach; achieved Dice is reported) plus **"wrong mask"**: the mask of a different case placed on this image. Question: can it detect even a grossly wrong outline?
- **D2 outline control.** The same slices with the red outline removed (and a blank projection panel) against the same slices with the correct outline, main question. Question: does the answer depend on the outline at all?
- **D3a counting.** Only the projection panel; "How many separate white shapes are there? Answer with one digit." Clean (1 piece) against extra island at Dice 0.9 / 0.8 / 0.6; score = log P("2") - log P("1"). Question: can it perceive a separate piece?
- **D3b outline perception.** One slice with and without the outline; "Is there a red line in this image? Answer YES or NO." Question: can it see a thin red line at all?
Predictions (fixed now): P1 wrong-mask AUROC < 0.8; P2 counting AUROC >= 0.8 at island Dice 0.8; P3 D2 outline vs no outline AUROC < 0.6 (the answer ignores the outline); P4 D3b >= 0.9 (it can see the line).
Branches, fixed now: (a) wrong-mask AUROC < 0.65 and D2 < 0.6 -> it does not use the outline for this question: stop prompting variants of this question, go to the "recognise what is inside the box" use or to a trained probe. (b) wrong-mask >= 0.8 but Dice 0.9/0.8 < 0.7 -> it sees gross errors only: try targeted questions and a few worked examples, then a probe. (c) anything in between: report as unclear and run D1 again with more cases. All tuning from here on uses cases 0-99 only; the test cases have already been scored by the VLM twice and will be used once more at the very end, said so in the report.

## Step 0 results (cases 0-29, n = 30 per side, wide intervals) and the departure from the pre-set rule
It sees the thin red line (AUROC 1.00), its answer depends on the outline (0.83), it ranks an extra island by counting shapes in the projection panel (0.91-0.93 even at Dice 0.9) but its argmax always says "1"; it does **not** judge outline quality (wrong-case mask 0.63; wrong size flat 0.57-0.59; shift only 0.69 at Dice 0.4; island only from Dice 0.6). Predictions P1, P2, P4 confirmed, P3 refuted. The pre-set rule gave branch (c) ("rerun D1 with more cases"); the intervals already exclude (a) (outline-distinguishability lower bound 0.70) and (b) (wrong-mask upper bound 0.74), so rerunning would only narrow them. **Departure, stated openly:** go straight to the action planned for "sees but cannot judge".

## Step 1: targeted questions (design, not yet run)
Idea: the one-size question "is the outline correct?" fails, but a perception question that matches one error ("how many separate shapes?") works. Test whether a matched question per error gives a useful check, and whether it adds anything to the simple checks.

**Pictures.** Questions about size and shift use the strip of the three MRI slices with the outline (first three panels of the four-panel view, no projection panel). The count question uses the projection panel alone, as in step 0.

**Questions, wording fixed now** (each answer is YES/NO; suspicion = log P(NO) - log P(YES). Preface for S1, S2, H1, H2: "The picture shows three slices of a brain MRI side by side. The red line is meant to outline one structure (the hippocampus).")
- S1 (too small): "Is the whole structure inside the red line, with no part of it sticking out beyond the line? Answer with exactly one word: YES or NO."
- S2 (too big): "Does the red line follow the edge of the structure closely, with no extra background included inside the line? Answer with exactly one word: YES or NO."
- H1 (shift): "Is the red line centered on the structure, not displaced to one side? Answer with exactly one word: YES or NO."
- H2 (shift, alternative): "Does the red line lie exactly on top of the structure it is meant to outline? Answer with exactly one word: YES or NO."
- C (island): the step-0 counting question on the projection panel; score = log P("2") - log P("1").
No other wordings will be tried.

**Matched score per error.** wrong size = max(z(S1), z(S2)); shift = z(H) for the wording chosen on the dev cases; island = z(C). z = (score - mean) / sd over the clean items of the dev cases (so the scales are comparable).

**Dev (cases 30-69, 40 cases; cases 0-29 were used in step 0).** Clean + 3 errors x Dice 0.9 / 0.8 = 7 versions per case, 5 calls each (S1, S2, H1, H2, C): about 1400 calls, 45-50 min. Uses: (1) pick H1 or H2 for shift, the only choice made on dev; (2) clean-item mean and sd for z; (3) a question-by-error AUROC matrix to see whether questions are specific.
**Test (cases 100-259, evaluated once, with the wording frozen).** The same 160 cases and damage as `run_main.py` / `run_vlm_main.py`; 4 calls per item (S1, S2, chosen H, C), about 4500 calls, about 2.5 h in the background (a 100-case version would take about 1.6 h). The test cases were already scored by the VLM twice (first round and the main run); this is the third use and the report says so.

**What is reported.** (1) AUROC with a case-bootstrap 95% CI per matched question and error; (2) the question-by-error matrix (specificity); (3) complementarity: z-score the volume check and the image check with the clean cases 0-99, take the max, and compare the AUROC of "simple checks only" with "simple checks + matched VLM score" per error and Dice, with a bootstrap CI on the difference.

**Predictions (fixed now; my expectation is that this only helps for the island)**
- T1. Island (count): AUROC >= 0.85 at Dice 0.9 and 0.8.
- T2. Wrong size: matched score AUROC < 0.70 at Dice 0.8.
- T3. Shift: matched score AUROC < 0.65 at Dice 0.9.
- T4. No matched VLM score beats the best simple check on any setting.
- T5. Adding the VLM score to the simple checks raises AUROC by less than 0.03 for size and for shift.
**Decision rules (fixed now).** A question "works" if AUROC >= 0.7 with the CI lower bound above 0.5 at Dice 0.9 for its error. If neither the size nor the shift question works on the test: stop prompting; a few worked examples are tried **only if** on dev at least one of them reached AUROC >= 0.6 (some signal); otherwise go to the trained probe on the model's internal representation (separate design). If one works: say so and report the complementarity result.
**Limits that stay.** One 3B model, zero-shot; the questions assume the structure looks dark grey in these crops (it does in the examples I looked at, not checked on all cases); the count question works only for an extra piece that is separate from the structure.

## Not in this version (shelved in `archive/DESIGN_v2_shelved.md`)
Learned CNN inspector and held-out error types, prompt-sensitivity re-runs, the slab error, a locked third data split, the check on unmodified labels, a second dataset, a larger VLM.

## Limits
One small dataset; synthetic errors are not real annotator errors; the real-error size (Dice) is not known; two simple checks only; no clinical claim.
