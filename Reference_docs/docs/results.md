# Results

*Updated only with the user's approval (rulebook R12). Last updated 2026-10-22 (section 7 completed with approval).*

## 1. Audit of the original MATLAB code and paper (Phase 0)
| # | Finding | Effect | Fixed in port |
|---|---|---|---|
| 1 | Hyperparameters chosen on the test set | Reported best accuracies are optimistically biased | Main protocol uses CV on train; test-selected kept as labelled comparison |
| 2 | τ = 1: upper bound (1−τ)c = 0 forces w = 0 → every sample predicted +1 | τ = 1 results depend on solver residue | Tanveer dual box −τc ≤ λ ≤ c |
| 3 | Linear branch drops the last test column | Crash (dimension mismatch) | Fixed |
| 4 | QP: linear term −τe, no lower bound | Not the published Pin-GTSVM dual | Correct dual (verified against primal) |
| 5 | Nearest-surface rule not divided by ‖w‖ | Biased class assignment | Normalised distances |
| 6 | False-positive list uses the last grid point | Wrong images analysed | Uses the selected model |
| 7 | Unseeded, non-stratified CV; no feature standardisation; NaN precision possible | Not reproducible | Seeded stratified CV; NaN reported explicitly; standardisation optional |
| 8 | Paper vs code: GoogLeNet layer, Table 6 vs Table 4 swap, batching/GPU claims | Inconsistent reporting | Documented |
| 9 | ResNet-50 features from `fc1000` (class scores) | Unusual feature choice | 2048-d pooled layer offered |
| 10 | MATLAB Caffe weights ≠ torchvision weights | Numbers cannot match exactly | Documented |

## 2. Port verification (Phase 1)
- 65/65 tests pass (`results/test_reports/2026-10-13_pytest.xml`); details in [test.md](test.md).
- Dual solution matches an independently solved primal to 1e-5 for linear and RBF kernels, τ ∈ {0, 0.5, 1}.

## 3. Solver cost (`results/benchmarks/2026-10-13_exact_solver_local.csv`)
Exact RBF solver, local CPU, random data, one hyperparameter setting:

| Problem | Size (pos + neg × dim) | Fit time |
|---|---|---|
| Hazy-parking scale | 380 + 150 × 4096 | 0.11 s |
| CNRPark-B scale | 4781 + 1632 × 1024 | 11.5 s |
| ImageNet-C one-vs-one pair | 2625 + 2625 × 768 | 6.1 s (11 s in an earlier run under load) |
| ImageNet-C one-vs-one pair | 2625 + 2625 × 1024 | 6.2 s |

→ 380 one-vs-one pairs (two 20-class sets × 190) ≈ 38–70 min per ViT per setting with the exact solver
(line updated 2026-10-13 with the user's approval; it previously referred to the superseded 100-class plan).

## 4. Environment (Phase 2)
torch 2.14.1+cu132 on RTX 3050 4 GB, CUDA 13.2, cuDNN 9.24; GPU double-precision matmul matches CPU to 2.4e-15
(`results/environment/2026-10-13_environment.json`).

## 5. Phase 6 — ImageNet-C data checks (Kaggle, CPU)
Data: Kaggle copy `vthutrang/imagenet-c` (third-party upload). Evidence: `results/phase6/phase6/` (run v1, 2026-10-13,
bundle v1) and `results/phase6/phase6_outputs_v2/phase6_v2/` (run v2, 2026-10-16, bundle v2).

| Check | Run v1 | Run v2 |
|---|---|---|
| Unit tests on Kaggle | 14 passed, 2 skipped | 16 passed, 2 skipped |
| Images in the two class sets (40 classes × 15 corruptions × 5 severities × 50) | 150,000 ✅ | 150,000 ✅ |
| Split by source image: 35 train / 15 test per class (5 CV folds on train) | 105,000 / 45,000 ✅ | 105,000 / 45,000 ✅ |
| Leakage (a source image in two splits or two folds) | none ✅ | none ✅ |
| Byte comparison with the official Zenodo `blur.tar` | failed: archive is gzip despite `.tar` → fixed (bundle v2) | failed: download cut off → resume fix (bundle v3) |

- `manifest.csv` of run v1 and run v2 are **byte-identical**, although the runs used different Kaggle machines (Intel
  Xeon, AMD EPYC): indexing and split are exactly reproducible. Run v1's manifest was also re-checked locally with 15
  independent checks, including rebuilding the seeded split (identical).
- Approved manifest fingerprint (SHA-256): `b3757fff48e907dca8a9d711f1421c76dacedfac38df69a4107a26f64e55136a`.
- **Limitation (user decision 2026-10-16):** the byte comparison with Zenodo was dropped; Phase 6 was closed on the data
  checks. Reports must state that the Kaggle copy was checked for completeness and structure, not byte-for-byte against
  the official release.

## 6. Phase 7 — frozen-backbone feature extraction (Kaggle, Tesla T4)
Evidence: `results/phase7/phase7_summary_v3/phase7_v3/` (definitive run v3, 2026-10-17: notebook
`phase7_features_v3`, bundle v5 `ec7db89`, timm 1.0.30, torch 2.11.0+cu128) and `results/phase7/phase7_summary_v1/phase7_v1/`
(interactive run v1, 2026-10-16: bundle v4, timm 1.0.29; its features were not kept). Figures with data tables:
`…/phase7_v3/figures/`.

**Setup.** All 7 backbones fully **frozen** (pretrained weights unchanged, no gradients, no head); 224 × 224 input
without resize/crop; fp16 autocast. Vectors: `global` = CLS token (DINOv2, DINOv3) or the model's pooled embedding
(SigLIP 2 attention pooling, FastViT); `patch_mean` = mean of patch tokens (ViTs only). FastViT in its deployed
(reparameterised) form. Checks passed: manifest fingerprint = Phase 6, timm 1.0.30, all 7 feature files saved
(150,000 rows each), sanity thresholds. Extraction time 192–602 s per backbone (≈ 45 min in total).

**Edge metrics** (one 224 × 224 image; GPU = Tesla T4 fp16, CPU = Kaggle Xeon 2.0 GHz, 2 threads, fp32):

| Backbone | timm weights | Params (M) | GMACs | GPU ms | CPU ms | Dims (global) |
|---|---|---|---|---|---|---|
| FastViT-SA12 | `fastvit_sa12.apple_in1k` | 10.52 | 1.48 | 7.27 | 40.3 | 1024 |
| FastViT-SA24 | `fastvit_sa24.apple_in1k` | 20.48 | 2.87 | 13.22 | 76.5 | 1024 |
| DINOv2-S/14 | `vit_small_patch14_dinov2.lvd142m` | 21.63 | 5.51 | 6.56 | 76.7 | 384 |
| DINOv3-S/16 | `vit_small_patch16_dinov3.lvd1689m` | 21.59 | 4.33 | 10.20 | 65.1 | 384 |
| DINOv2-B/14 | `vit_base_patch14_dinov2.lvd142m` | 85.72 | 21.94 | 7.66 | 239.5 | 768 |
| DINOv3-B/16 | `vit_base_patch16_dinov3.lvd1689m` | 85.64 | 17.19 | 10.32 | 204.5 | 768 |
| SigLIP2-B/16 | `vit_base_patch16_siglip_224.v2_webli` | 92.88 | 17.00 | 7.20 | 202.6 | 768 |

**Sanity accuracy** (severity 1 only; standardise + logistic regression, train → test split; a check that each
extractor works, not the study's method or result):

| Backbone | Similar-20 global | Similar-20 patch_mean | Dissimilar-20 global | Dissimilar-20 patch_mean |
|---|---|---|---|---|
| FastViT-SA12 | 0.751 | — | 0.950 | — |
| FastViT-SA24 | 0.787 | — | 0.952 | — |
| DINOv2-S/14 | 0.741 | 0.630 | 0.969 | 0.972 |
| DINOv3-S/16 | 0.722 | 0.572 | 0.966 | 0.967 |
| DINOv2-B/14 | 0.789 | 0.660 | 0.974 | 0.978 |
| DINOv3-B/16 | 0.788 | 0.717 | 0.979 | 0.978 |
| SigLIP2-B/16 | 0.666 | 0.640 | 0.976 | 0.972 |

Observations (to be tested properly in Phases 10–11):
- Dissimilar-20 is near ceiling for every backbone at severity 1 (0.95–0.98); Similar-20 separates them.
- The test set is 300 source images × 15 corruptions; rows sharing a source are not independent, so differences below
  about 3 points are not evidence of a real difference. FastViT-SA24, DINOv2-B and DINOv3-B are tied (0.787–0.789);
  SigLIP2-B is about 12 points lower on Similar-20 while among the best on Dissimilar-20.
- `global` beats `patch_mean` on Similar-20 by 3–15 points and equals it on Dissimilar-20.
- **Confound to disclose:** FastViT (`apple_in1k`) was trained with ImageNet-1k labels, which include all 40 classes
  (training images only; ImageNet-C is built from the validation images, so there is no image overlap). DINOv2/DINOv3
  are self-supervised and SigLIP 2 is image–text trained.
- Batch-1 GPU latency is dominated by per-layer overhead, not arithmetic (S and B ViTs within ~1 ms); CPU latency
  follows compute. Reparameterisation cut FastViT latency from 12.9 → 7.3 ms (SA12) and 24.0 → 13.2 ms (SA24) on GPU,
  61 → 40 ms and 105 → 76 ms on CPU, with unchanged features.
- Reproducibility across runs: sanity accuracy of run v3 equals run v1 to 4 decimals for DINOv2, DINOv3 and SigLIP 2
  (so timm 1.0.29 → 1.0.30 had no effect); FastViT differs by ≤ 0.0016 (reparameterised form, fp16 rounding).

**Run history.** Committed runs of notebooks v1/v2 never started ("Cancelled after 0 seconds": queued behind other GPU
sessions). Fixed between runs: FLOP-counter crash on SigLIP 2's attention pooling (frozen weights), timm version check
before the first import, FastViT reparameterisation, live progress, per-backbone 30-min and 5-min stall limits.

## 7. Phase 8 — a Pin-GTSVM solver fast enough for the grid, and exact
Question: the paper's exact solver (CVXOPT, full RBF kernel) takes ~17 s per class pair and ~5.5 s per extra (c, τ)
setting on Kaggle's 4-core CPU — far too slow for 380 pairs × a hyperparameter grid × CV. Which faster solver gives
the **same predictions**? Pass mark (approved 2026-10-17, never relaxed): ≥ 99 % identical test predictions and
≤ 0.5 points accuracy difference against the exact method, on all 190 pairs of both class sets at σ = median × {¼, 1, 4}
(c = 1, τ = 0.5) plus 10 pairs per class set at corner settings. Checked on two feature configurations: DINOv3-S/16
(384-d) and FastViT-SA24 (1,024-d), global vector, standardised with the training split.

| Check | Where | Solver | Result |
|---|---|---|---|
| v1 (2026-10-17) | Kaggle CPU, bundle v6 | Reduced kernel, **one landmark set shared by all 20 classes** (100 / 300 / 1,000) | ❌ best (1,000): 97.6 % / 97.1 % identical (DINOv3), 95.9 % / 96.2 % (FastViT); only 1 in 10 landmarks belongs to a pair's classes |
| v2 | — | Per-class landmarks, exact reference on a P100 | not run: Kaggle no longer offers a P100 (2026-10-18) |
| v3 (2026-10-18/19) | Kaggle CPU, bundle v8 | **Per-class landmarks** (50 / 150 / 500 per class); reference = the paper's method via CVXOPT; corners widened to the paper's grid extremes (c ∈ {2⁻⁵, 2⁻³, 1, 2⁵} × τ ∈ {0, 0.5, 0.8, 1}) | ❌ DINOv3 best (500): 98.48 % / 97.96 % identical, −0.42 / −0.35 points; FastViT 150: 96.7 % / 93.7 %, the 500 run crashed (interior point, not positive definite). Only ~4× faster than exact. **CVXOPT predictions saved for all checks of both models** |
| v4 (2026-10-20) | Kaggle T4, bundle v9 | The **exact** method in torch, float32 | crashed in its 2nd unit: float32 H'H + εI not positive definite (rounding ≈ 0.3 > ε = 0.05); float32 interior point short of tolerance for 356 / 380 QPs |
| **v5 (2026-10-20)** | **Kaggle T4, bundle v10, 1.44 h** | **Exact method, fp32qr**: factor from a float32 QR of [H; √ε I] (H'H never formed), float32 interior point, re-solved in float64 when not converged | ✅ **both models: 99.998 % (all pairs) / 99.98 % (corners) identical, accuracy difference ≤ 0.003 points**; fp64 on the corners: 100 % identical |

Timing on the T4 (check v5), per class pair: build 0.53–0.55 s (fp32qr) vs 3.4–3.6 s (fp64); solve per (c, τ) setting
0.66 s (DINOv3) / 1.35 s (FastViT) for fp32qr, ~0.95 s for fp64 — against ~11 s + ~5.5 s on the CPU with CVXOPT.
A few QPs ended at the iteration limit (DINOv3 7, FastViT 49, of ~2,900 QPs per model) without changing the predictions; the limit of
the float64 re-solve was raised from 100 to 200 iterations in bundle v11 — a mistake, see check v6.

Check v6 (2026-10-21, T4, bundle v11, 3.46 h, stopped by its time limit): the **batched** solver (many QPs per GPU call)
gave the same predictions as CVXOPT (DINOv3: 99.998 % / 99.98 % identical, passed; FastViT: 99.998 % / 99.984 %, its
second corner unit cut by the time limit) but was **slower** than v5 — 1.67 s (DINOv3) and 6.83 s (FastViT) per pair
and setting instead of 0.66 / 1.35 s — and the cost benchmark of the grid did not run. Causes: the batched solver kept
factorising problems that had already converged while a few problems that never converge ran up to 200 float64
iterations; and at n = 2,625 the T4 is already compute-bound, so batching saves little (the expected 3–10× gain was an
unmeasured estimate). Fixed in bundle v12: the batch shrinks to the unsolved problems, a problem that stops improving
is stopped, the float64 re-solve is capped at 60 iterations, and each unconverged QP is logged (check v7).

**Check v7 (2026-10-22, T4, bundle v12, 0.52 h) — Phase 8 closed.**
- Re-check of the units that were slow or missing in v6, against the CVXOPT predictions: all passed — FastViT all pairs
  Dissimilar-20 σ × ¼: 99.993 % identical (−0.002 points); FastViT corners Dissimilar-20 / Similar-20: 99.989 % / 99.982 %;
  DINOv3 corners Dissimilar-20: 99.984 % (+0.006 points).
- Speed of the fixed batched solver on those units: 1.86 s (FastViT σ × ¼), 1.61 s and 0.63 s (FastViT corners) and
  0.89 s (DINOv3 corners) per pair and setting — faster than v5 (3.45, 3.18, 0.71, 1.36 s) and v6 (25.0, 19.0, —, 9.1 s).
- 88 of the 4,220 QPs ended without reaching the tolerance (87 stopped as stalled, the rest at the iteration limit; final complementarity ≤ 1e-5);
  none changed the pass mark. All have τ > 0 (none τ = 0); mostly c = 32 or σ × ¼, mostly plane 1, and three quarters
  involve Dissimilar-20 classes 0 (great white shark), 18 (yellow lady's slipper) or 19 (agaric). Logged in `unconverged_qps.csv`.
- **GPU cost of the approved tuning protocol** (fold size, measured on 5 of the 20 tuning pairs, extrapolated; final
  fits as an upper bound): DINOv3 0.57 h (Similar-20) / 0.89 h (Dissimilar-20); FastViT 0.79 h / 1.40 h — mean
  **0.91 T4 hours per model and class set, ≈ 12.8 h for all 7 models × 2 class sets** (Kaggle quota: ~30 h per week).

Evidence: `results/phase8/phase8_check_v6/`, `results/phase8/phase8_check_v7/`.

Conclusions:
- The landmark (reduced-kernel) shortcut is **not** used: it never reached 99 % identical predictions, and where it came
  close it was only ~4× faster than the exact method.
- The classifier of all later phases is the **paper's exact Pin-GTSVM** (full RBF kernel, corrected dual, ε = 0.05),
  solved on a Kaggle T4 in fp32qr — validated against CVXOPT, not an approximation.
- Evidence: `results/phase8/` (v1: `phase8_*_v1/`; v3: `phase8_check_v3_dinov3_s16/`, `results_failed/` (FastViT,
  with its log); v5: `phase8_check_v5/`); reference predictions as a Kaggle dataset: `kaggle/dist/phase8_references_v1.zip`.

Grid for screening and the main runs (approved 2026-10-20; the original 180-setting grid with 5-fold CV would need
~30 T4 hours per model and class set): σ = median training distance × {¼, ½, 1, 2, 4}; c ∈ {2⁻⁵, 2⁻³, 2⁻¹, 2, 2³, 2⁵}
(the paper's range); τ ∈ {0, 0.5, 0.8, 1} (the paper's values plus the hinge case τ = 0 for RQ2); validation on 3 of
the 5 predefined CV folds of the Phase 6 split. Tuning protocol (approved 2026-10-21): **shared** hyperparameters
(one σ, c, τ for all pairs of a model and class set, as LIBSVM does for one-vs-one) chosen on **20 seeded class pairs**
per class set; then all 190 pairs are fitted at the chosen setting and at the best τ = 0 setting (RQ2). Check v7
measures the GPU cost of this protocol before it runs.

## 8. Experiments
None yet.
