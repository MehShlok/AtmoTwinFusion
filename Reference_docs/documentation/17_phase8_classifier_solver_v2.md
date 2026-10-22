# Phase 8: making the classifier fast enough — without changing it

*Version 2 · 2026-10-22 · supersedes `17_phase8_classifier_solver_v1.md` (kept unchanged). Evidence: `../results/phase8/`. Technical record: `../docs/results.md`, section 7.*

**What changed from v1:** the last two checks (v6, v7), the approved tuning protocol and the measured GPU cost — Phase 8 is finished (section "How it ended" below).

## In one paragraph
The study's classifier is the paper's Pin-GTSVM. Solved the paper's way (CVXOPT on a CPU), one class pair takes about
17 seconds, and every extra hyperparameter setting about 5.5 seconds more. The planned experiments need hundreds of
thousands of such solves, which would take months. Phase 8 looked for a faster way that gives **the same answers**.
The first idea — a shortcut that approximates the classifier — failed the accuracy check four times. What passed is
the **exact** method of the paper, computed on Kaggle's T4 graphics card with a numerically careful recipe: it agrees
with the original on 99.998 % of about 6.6 million test predictions, and it is 4–14× faster.

## How "the same answers" was checked
Before any run we fixed a pass mark (approved 2026-10-17): the faster solver must give **identical predictions for at
least 99 % of the test pictures** and an accuracy **within 0.5 points** of the original, on all 190 class pairs of both
class sets, at three kernel widths, plus 10 pairs per class set at extreme settings — including the extremes of the
paper's own settings (c from 2⁻⁵ to 2⁵; τ = 0.5, 0.8, 1) and τ = 0. It was checked on two very different feature sets
(DINOv3-S: 384 numbers per picture; FastViT-SA24: 1,024). The pass mark was never relaxed.

## What was tried, and why it failed or passed
| Attempt | Idea | What happened | Why |
|---|---|---|---|
| 1 (CPU) | Compare each picture with a few hundred "landmark" pictures shared by all classes, instead of all 5,250 | ❌ at best 97.6 % identical | Only 1 in 10 landmarks belonged to the two classes being separated |
| 2 | The same with landmarks per class, checked on a P100 graphics card | not run | Kaggle no longer offers the P100 |
| 3 (CPU) | Landmarks per class (up to 500 each); the original solver's answers saved as the reference | ❌ at best 98.5 % identical, and only ~4× faster | Close boundaries — similar terriers, narrow kernels, weak regularisation — need nearly all pictures |
| 4 (T4) | The exact method on the graphics card in fast "single precision" | ❌ crashed | Single precision rounds the key matrix by about 0.3, more than its safety margin of 0.05, so it stops being solvable |
| **5 (T4)** | **The exact method, with a numerically safe recipe** | ✅ **99.998 % identical, accuracy equal** | See below |

*Why the shortcut was given up rather than tuned further:* where it came close to the pass mark it was hardly faster than
the exact method, so it bought little and would have added an approximation that every result would have to defend.

## The recipe that passed, in plain words
- **Never square the problem.** The original formula multiplies a large matrix by itself, which in single precision
  loses the small details that matter. The new recipe uses a "QR decomposition" that gets the same factor without that
  multiplication, so single precision is enough for the bulk of the work.
- **Finish in double precision when needed.** Each optimisation runs first in fast single precision; if it has not
  reached the required accuracy after 30 steps, it is solved again in slower but exact double precision.
- **Never stop on a failure.** If any precision fails, the notebook records it and moves on, so a GPU run is never
  wasted by one error.

Result: identical predictions on 99.998 % (all pairs) and 99.98 % (extreme settings) for both models; in full double
precision the extreme settings were 100 % identical.

## What it means for the study
- The classifier in every later result is **exactly the paper's method** — no approximation to disclose.
- The study needed a smaller hyperparameter grid to fit Kaggle's GPU quota. You approved (2026-10-20): kernel width at 5
  values (median distance × ¼, ½, 1, 2, 4), c at 6 values across the paper's range, τ at the paper's 0.5, 0.8, 1 plus
  0 (the comparison without the pinball loss), and validation on 3 of the 5 prepared folds.
- Next (check v6): solving many optimisations at once on the GPU ("batching"), checked against the same saved answers,
  and a measurement of how many GPU hours the main experiments will need.

## How it ended (checks v6 and v7)
- **Check v6** solved many optimisations together. The answers were still identical, but it was *slower*: it kept
  working on problems that were already solved while a few hard ones never finished, and the run ran out of time.
  *Lesson:* the graphics card was already fully busy with one problem at a time, so batching could not help much; my
  estimate of a 3–10× gain had not been measured.
- **Check v7** fixed it: finished problems leave the batch, a problem that stops improving is stopped, and the slow
  double-precision repair is limited. The units that had been slow passed again (99.98–99.99 % identical) and ran
  faster than ever.
- About 2 % of the optimisations (88 of 4,220) stop just short of full accuracy. They never changed a result. They all
  use the pinball loss (τ > 0), mostly with the weakest regularisation (c = 32), and three quarters involve three of
  the "unrelated" classes (among them a mushroom and an orchid) — recorded for the write-up.

## The tuning plan and its cost (your decision, 2026-10-21)
- **One setting for all class pairs** of a model and class set (σ, c, τ), as the widely used LIBSVM library does for
  one-vs-one. It is chosen on **20 randomly drawn (but fixed) class pairs** instead of all 190, using 3 of the prepared
  validation folds; then all 190 pairs are trained at the chosen setting, and once more without the pinball loss
  (τ = 0) for the comparison the study is about.
- **Measured cost:** about 0.6–1.4 hours on a T4 per model and class set — about **13 hours for all 7 models and both
  class sets**, within one week of Kaggle's ~30-hour GPU quota (the full 190-pair search would have needed ~100 hours).
- **Next:** the screening of all 7 models (Phase 8b) and the main experiments (Phase 10).

## Terms
- **Pass mark / gate:** the accuracy rule a faster method must meet before it may replace the original.
- **Single / double precision (float32 / float64):** how many digits a computer keeps; the T4 does single precision
  about 30× faster than double.
- **Landmarks / reduced kernel:** comparing each picture with a sample of training pictures instead of all of them.
- **Batching:** solving many optimisations in one call to the graphics card.
- **Validation fold:** part of the training pictures held back to choose settings; the test pictures are never used for it.
