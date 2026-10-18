# Design v2.1 (protocol, written before the v2 experiments)

Status: design only. Facts marked **[measured]** were checked on the data; **[to verify]** are assumptions that a pilot must confirm. Nothing here is a result of v2.

## 1. Question
For an (MRI, mask) pair, which mask errors can which inspector detect, **as a function of error size**? Can a learned inspector cover errors that simple rules miss, including error types it never saw in training? Is a zero-shot VLM useful at all?

## 2. Why v2 (what v1 got wrong)
- Severity was set in voxels. The structure is about 9 voxels at its thickest (median 8.7, range 6.9-10.4) **[measured]**, so 1 voxel of erosion removes 41% of the volume (Dice 0.74) and 3 voxels removes 89%; dilation adds 48-163%; slab/blob at severity 1 change Dice by only 0.01 **[measured]**. Difficulty was bimodal: erode/dilate trivial for a volume check, slab/blob invisible.
- The three VLM slices contained 7-17% of the changed voxels (7% for the blob; a random voxel set would give 7.7%) **[measured]**.
- CNN training used severity 1 as "corrupted" although it is almost identical to clean (label noise).
- The test cases were reused while the edge detector was fixed, so they are no longer a clean test.
- Erosion/dilation via the distance transform moves in coarse steps (one voxel shell = about 11% of the volume), so a Dice target cannot be hit **[measured]**. A level-set method fixes this (section 4).

## 3. Data and splits (by case)
MSD Task04 Hippocampus, 260 cases, labels merged to one binary mask (CC-BY-SA 4.0; public data, no clinical claim).
| cases | use |
|---|---|
| 0-79 | CNN training |
| 80-99 | CNN validation (early stopping) |
| 0-99 | reference set for calibrating the rule inspectors (they use no labels, only clean-mask statistics; 20 cases would be too few) |
| 100-159 | dev set: pilots and tuning only |
| 160-259 | **final test, evaluated once** after every choice is frozen |
Every mask, clean or corrupted, passes through the same Gaussian smoothing (sigma 0.7) so that morphology artefacts cannot be a shortcut.

## 4. Error generation: severity = overlap with the clean mask
Target Dice in {0.95, 0.90, 0.80}, the same for every error type. A continuous parameter per case is found by bisection.
| error | continuous parameter | status |
|---|---|---|
| erode / dilate | threshold of the Gaussian-smoothed mask (level set) | **[measured]** Dice hit exactly for 100% of 60 cases; volume change erode -10/-18/-33%, dilate +11/+22/+50% |
| shift | sub-voxel shift (linear interpolation, threshold 0.5), random 3D direction | **[to verify]** |
| drop_slab | soft cut along the longest axis from one end | **[to verify]** |
| spurious_blob | island volume V_b = 2V(1/Dice - 1): 10%, 22%, 50% of V; random location outside the structure | **[to verify]**; at Dice 0.8 the island is half the structure, extreme and unrealistic, report it as such |
Population volume CV is 13.1% **[measured]**, so a volume-only inspector should find a -33% erosion but struggle with -10%: difficulty is now graded.

## 5. Inspectors
1. blind: robust z of mask volume (no image)
2. edge: boundary gradient, two-sided deviation
3. components: connected components minus one
4. rules: max(edge, 3 x components)
5. learned 3D CNN: 2 channels (MRI, mask); trained on online corruptions with target Dice drawn uniformly from [0.78, 0.97] (nothing closer to clean than 0.97); models trained on all types and with each type held out; shuffled-label control; 3 seeds; flips, joint shifts, intensity jitter; small network, dropout, weight decay, early stopping. Option to add a gradient channel is decided on dev, then frozen.
6. VLM, zero-shot, Qwen2.5-VL-3B, 3 fixed prompts. Views: orthogonal slices at the centroid of the largest component, plus a projection of the mask outline (so a distant island is visible), plus an unannotated copy for comparison. No view uses knowledge of where the error is.

## 6. Evaluation protocol
- Paired items: each test case contributes one clean item and one corrupted item per (type, Dice). AUROC of corrupted vs clean per (inspector, type, Dice).
- 95% CI by bootstrap over **cases** (resample cases, recompute AUROC).
- Primary figure: AUROC vs Dice curve per error type and inspector. Secondary: the held-out-type CNN versus the all-type CNN.
- Rule-vs-CNN comparisons use the same items.

## 7. Pre-registered success criteria (thresholds are conventions, not derived; fixed now)
With 100 test cases the CI half-width is about 0.08, so every "AUROC >= x" criterion also needs the **lower CI bound > 0.5**.
- Rules reproduce "each rule has a type it catches": AUROC >= 0.8 for its own type at Dice 0.9. Otherwise the v1 finding was an artefact of severity.
- CNN useful: on **held-out types**, AUROC >= 0.75 at Dice 0.8 on at least 2 types, and better than the best rule on at least one type the rules miss. Otherwise report "not better than rules".
- Shuffled-label control within 0.5 +/- 0.1; otherwise there is a leak and the results are discarded.
- VLM: pilot on dev (20 clean + 20 corrupted per type); full run only if the pilot AUROC > 0.6 for at least one type. Final success: AUROC >= 0.65 at Dice 0.8 on at least 3 of 5 types and prompt agreement (Spearman) >= 0.6. If not met: "not usable zero-shot", stop, **no more prompt hunting**. A 7B model only if the 3B pilot shows a signal.
- 5 types x 3 Dice levels x 6 inspectors is many comparisons; only the pre-registered criteria are treated as confirmatory, everything else is descriptive.

## 8. Pilots (go / no-go before any full run)
- P1: Dice targets hit within 0.01 for >= 95% of cases, for every error type (shift, slab, blob still to verify).
- P2: look at 5 overlays per Dice level and error type; 0.95 should be barely visible, 0.8 clearly visible; otherwise change the targets.
- P3: VLM pilot (section 7).
- P4: CNN train/validation gap and the shuffled control on dev.

## 9. Real-data check
Synthetic errors are not annotator errors. Run the best inspector on the **unmodified** labels (all 260) and look at the 10 most suspicious overlays; write down what is wrong with each, if anything. Qualitative only: there is no ground truth, so no precision is claimed.

## 10. Known limits
One dataset (small crops, 1 mm voxels); synthetic errors; 3B model only; centroid-based views are an approximation of what a human reviewer sees; no clinical validity; no longitudinal part.

## 11. v1 numbers this design builds on (for reference, all measured)
- Rules vs the CNN's own validation set (20 cases, clean vs 5 types x severity 2-3): blind 0.72, edge 0.69, rules 0.81; first-round CNN validation AUROC 0.58-0.70.
- Zero-shot 3B VLM: answered NO on 700/700 images, AUROC 0.27-0.50, prompt agreement Spearman 0.23-0.57.
