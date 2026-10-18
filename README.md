# Mask audit: which inspector catches which segmentation error?

**Question.** Given a (MRI, segmentation mask) pair, can an automatic inspector tell that the mask is wrong, and which kinds of errors does it miss?
This is the image version of a data-QC loop: inject errors with known ground truth, score each inspector, then look at *which* errors slip through.

**Data.** Medical Segmentation Decathlon, Task04 Hippocampus (CC-BY-SA 4.0), 260 cases, anterior+posterior labels merged into one binary mask. Public data only. Not clinical; this is a method demo.

**Errors injected** (severity 1-3 voxels): `erode`, `dilate`, `shift`, `drop_slab` (remove a slab at one end), `spurious_blob` (extra separate island).

**Inspectors** (score = suspicion, higher = more likely wrong; calibrated on 100 clean reference cases, tested on disjoint cases)
| name | sees the image? | what it does |
|---|---|---|
| `blind` | no | robust z-score of mask volume only (analogue of a text-only baseline) |
| `edge` | yes | mean image gradient on the mask boundary, deviation from normal in either direction |
| `components` | no | number of connected components minus one |
| `rules` | yes | max(edge, 3 x components) |
| `vlm` | yes | Qwen2.5-VL-3B-Instruct (local, bf16, Apple M4/MPS) sees 3 orthogonal slices with the red outline, answers YES/NO; score = log P(NO) - log P(YES) |

## Results (AUROC, corrupted vs clean; 0.5 = chance)

Classical inspectors, full test set (160 cases), 2 different reference/test splits (seeds 1 and 2); ranges are min-max over splits and severities, splits differ by up to about 0.07:

| error | blind | edge | components | rules |
|---|---|---|---|---|
| erode (s1-3) | 0.99-1.00 | 0.99-1.00 | 0.73-0.98 | 0.99-1.00 |
| dilate (s1-3) | 0.95-1.00 | 0.68-0.87 | 0.47-0.48 | 0.66-0.85 |
| shift (s1-3) | 0.50 | 0.58-0.72 | 0.50 | 0.56-0.70 |
| drop_slab (s1-3) | 0.49-0.75 | 0.48-0.63 | 0.50 | 0.48-0.62 |
| spurious_blob (s1-3) | 0.50-0.63 | 0.48-0.54 | 0.90-0.97 | 0.90-0.96 |

Reading: each error type has a different "detector that sees it". Volume-only catches size changes but is blind (0.50) to a shifted mask or an extra island, because the volume does not change. Connected components catch islands only. Boundary edge catches shifts only partially. **Subtle errors (small shifts, a thin slab missing) are caught by nothing here** (best AUROC about 0.7). A real QC system needs several inspectors plus the error taxonomy.

VLM inspector, 100 clean vs 60 per (error, severity 2 and 3), same items as the baselines:

| error | vlm s2 | vlm s3 | best classical s3 |
|---|---|---|---|
| dilate | 0.39 | 0.45 | 1.00 (blind) |
| drop_slab | 0.42 | 0.49 | 0.70 |
| erode | 0.40 | 0.27 | 1.00 |
| shift | 0.49 | 0.50 | 0.73 |
| spurious_blob | 0.47 | 0.48 | 0.88 (combined) |

**The 3B VLM is not a usable inspector in this zero-shot setup.** It answered NO on 100% of the 700 images; scores per group differ by <0.1 around 0.75 (clean 0.77 +/- 0.10). The answer format is fine (99% of probability mass on YES/NO), so this is not a parsing problem.

Prompt sensitivity (3 wordings, 60 clean + 30 per error at severity 3; `results/prompt_sensitivity_*.csv`): AUROC stays at or below chance for almost every error under all three prompts (0.20-0.53), and item-level scores correlate only weakly between prompts (Spearman 0.23-0.57). So the wording changes the ranking more than the image does. Eroded masks are scored as *more* acceptable (AUROC 0.20-0.37), the opposite of useful.

## Things that went wrong or were changed (kept honest)
- The first edge detector assumed "weak boundary edge = suspicious". On real MRI that is wrong for `dilate` (AUROC 0.10-0.25): the gradient keeps rising 1-3 voxels *outside* the labelled boundary (mean 0.101 at 0, 0.115 at +3), because the hippocampus touches strong edges (CSF, neighbouring structures). I changed `edge` to two-sided and kept the old one as `edge_signed` in the results. This choice was made after seeing the test numbers, so the splits are not fully independent evidence; the pool is only 260 cases.
- `spurious_blob` is nearly invisible to the VLM by construction: the three slices go through the mask centre and usually do not cut the island. A human reviewer looking at the same picture would also miss it.
- Synthetic-noise errors are not real annotator errors. Next step would be to hand-check the top-N alarms on unmodified data.

## Not done / limits
- Only the 3B model was tested. A 7B model (about 16 GB download) or a medical VLM may do better; not tried.
- No longitudinal part (needs ADNI/OASIS access). Second domain (road damage, e.g. RDD2022) not started and its availability is not verified.
- VLM subsample: 700 images, one seed.

## Reproduce
```
python3 run_synthetic.py              # sanity check, no downloads
python3 run_msd.py 0                  # classical inspectors on MSD (seed = split)
python3 run_vlm.py --n_bad 60         # VLM vs baselines (needs the model, ~1.5 s/image on M4)
python3 run_prompt_sens.py            # prompt sensitivity
```
Data goes in `data_raw/msd_hippocampus/` (Task04_Hippocampus.tar from the MSD AWS mirror). The model is cached by Hugging Face; delete `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct` when no longer needed (7.5 GB).
