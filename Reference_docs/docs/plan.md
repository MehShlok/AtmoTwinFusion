# Plan

*Living document — last updated 2026-10-18. One phase at a time; each ends with results, verification and the user's go-ahead.*

**Workflow principle:** test combinations *thoroughly* on **subsets of ImageNet-C** (now: the two 20-class sets) to save
time — smaller data, never fewer checks — record every change, pick the best configuration, then scale-up test it on more
data; least system complexity and edge deployment in mind. **All runs on Kaggle.**

| # | Phase | Status | Verification gate |
|---|---|---|---|
| 0 | Read paper and MATLAB code; audit | ✅ done | 10 findings recorded (results.md §1) |
| 1 | MATLAB → corrected Python port (`python_port/`) | ✅ done | 65/65 tests; dual = primal to 1e-5 |
| 2 | Environment: venv, torch 2.14.1+cu132 | ✅ done | GPU check, `pip check`, all tests incl. feature layers |
| 3 | Docs + `results/`; class sets Similar-20 / Dissimilar-20 (`study/`) | ✅ done (approved 2026-10-13) | 9/9 study tests (selection reproducible); docs approved |
| 4 | **Task 4** — deep research: best ViTs for ImageNet-C as feature extractors, incl. edge efficiency | ✅ done ([research_vits.md](research_vits.md)); 7 backbones approved | Cited shortlist; user approval |
| 5 | **Task 5** — research that challenges the approach | ✅ done ([research_counterevidence.md](research_counterevidence.md)); A1–A4, A8 approved | Summary of counter-evidence; user review |
| 6 | Data on Kaggle: manifest, grouped 70/30 split + CV folds | ✅ done (runs v1 + v2: manifest, split, leakage passed; manifests identical; re-verified locally). Zenodo byte comparison dropped by the user 2026-10-16 (notebook v3 kept, not run) | Manifest + leakage test pass |
| 7 | Feature extraction + cache (frozen ViTs first; Kaggle GPU) with edge metrics | ✅ done 2026-10-17 (run v3, bundle v5; features = output of the user's Kaggle notebook `phase7_features_v3`; results.md §6) | Manifest fingerprint = Phase 6; shapes, finiteness, metadata; severity-1 sanity accuracy |
| 8 | Fast Pin-GTSVM (reduced kernel + fast QP) + one-vs-one + minimal inference format | ✅ check v1 (CPU, shared landmarks) failed the gate; check v2 needed a P100, which Kaggle no longer offers (2026-10-18); check v3 (CPU, 2026-10-18/19): landmark shortcut failed (best 98.5 % identical at 500 per class, only ~4× faster than exact); CVXOPT reference predictions saved for both models. Check v4 (GPU T4) crashed (float32 H'H + eps I not positive definite). ✅ **Gate passed (check v5, 2026-10-20): the paper's exact method on a T4 in fp32qr, 99.998 % identical to CVXOPT for both models** (results.md §7). Check v6 (batched) was exact but slower and ran out of time. ✅ **Check v7 (2026-10-22): fixed batched solver passed the re-check; approved tuning protocol ≈ 0.9 T4 h per model and class set, ≈ 12.8 h for all 7 models — Phase 8 done** | ≥ 99 % identical predictions and ≤ 0.5 points accuracy difference vs the exact solver on all 380 pairs; unit tests |
| 8b | **Screening** (approved 2026-10-17): all 7 ViTs × all corruptions/severities × both class sets; linear probe, kNN, Pin-GTSVM at one default setting (Kaggle CPU) | ⏳ | Drop rule fixed in advance (scope.md): dropped if another ViT is not significantly less accurate and at least as fast on CPU |
| 9 | Experiment framework: config matrix, run registry, selection report | ⏳ | Tests: matrix expansion, cache keys, append-only registry |
| 10 | Frozen-ViT runs on the subset (screening survivors only): both class sets, all 15 corruptions × 5 severities, grouped CV, baselines (τ = 0, SVM, linear probe, kNN), L2 vs standardisation, standard + held-out-family tests | ⏳ | Registry complete; metrics per corruption, severity and family; Demšar statistics |
| 11 | Fine-tuned-ViT runs, same protocol — **top 4 survivors** | ⏳ | Training logs; no test-split use during fine-tuning; same checks as 10 |
| 12 | **Task 3** — 2D/3D representation and geometry analysis | ⏳ | Faithfulness measures with every figure |
| 13 | Choose the scale-up configuration; scale-up test plan | ⏳ | Selection report + user decision |
| 14 | Audit and write-up | ⏳ | Independent re-run of a sample of results |

## Phase details

### 4 — ViT research (next)
Sources: papers (arXiv, Google Scholar), ImageNet-C benchmark tables, model hubs (timm, Hugging Face), Kaggle.
Output: a shortlist across ViT families and sizes — including small, edge-friendly models — with, for each:
ImageNet-C numbers, pretraining data (and any overlap risk with ImageNet validation images), feature layer, input size,
parameters, MACs, measured or reported latency, licence, and whether it fine-tunes on a Kaggle T4.

### 5 — Counter-evidence
E.g. limits of ImageNet-C (augmentation–corruption similarity), pretraining-data contamination, kernel methods vs simple
linear probes on ViT features, weaknesses of pinball loss (loss of sparsity), twin-SVM criticisms, edge-deployment costs
of ViTs.

### 6 — Data
Kaggle notebook: index `vthutrang/imagenet-c` for the 40 approved classes, check counts and names, compare a random
sample byte-for-byte with the official Zenodo archives, write the manifest and the 70/30 split by source image.

### 7 — Features and cache
One extraction per (ViT, weights, frozen/fine-tuned) covering all 40 classes × 75 corrupted versions; every data-subset
combination is then a row filter on cached features. Edge metrics (parameters, MACs, size, latency, memory) recorded once
per ViT.

### 8 — Fast solver (approved 2026-10-13; settings approved 2026-10-17)
Exact cost ≈ 6–11 s per pair (measured) → 38–70 min per setting for 380 pairs. Design: reduced (rectangular) kernel with a
global landmark set per class set (Lee & Mangasarian 2001) + first-order box-QP solver (Beck & Teboulle 2009), checked
against CVXOPT; validated on all 380 pairs. Inference artifact = ViT + landmarks + surface weights as plain arrays.
- Landmarks: one shared set per class set, drawn from the training split only, balanced over classes and corruptions;
  sizes 100 / 300 / 1,000 tested, the smallest that passes is kept.
- Grid (180 settings): c ∈ {2⁻³ … 2⁵} × τ ∈ {0, 0.1, 0.5, 1} × σ = median training distance × {¼, ½, 1, 2, 4}.
- Steps: (1) solver + unit tests locally (optimality conditions, CVXOPT agreement on small problems, all-landmarks case
  = exact method); (2) Kaggle CPU: fast vs exact on all 380 pairs for 2 feature configurations (one 384-d, one 1024-d);
  (3) Kaggle GPU (~15 min): measured fit time → exact GPU-hour budget for 8b, 10 and 11 before they run.

### 8b — Screening (approved 2026-10-17)
Cheap, complete comparison of all 7 ViTs before the expensive grid; drop rule in scope.md. Survivors go to Phase 10;
the top 4 go to Phase 11. Features of dropped ViTs stay on Kaggle, so a ViT can be re-added.

### 9–11 — Framework and runs
One YAML defines the axes; the runner expands combinations, reuses cached features, and appends every run to
`results/registry.csv` with what changed. Time is saved by the amount of data (class subsets, or a data-fraction axis
for even quicker iterations), never by dropping checks: every run uses the same complete protocol. Code reaches Kaggle
as a small dataset (or repository) holding `python_port/` and `study/` — chosen in Phase 6.

### 12 — Representation analysis (methods decided after research)
Candidates: PCA, LDA, t-SNE, UMAP, PaCMAP; faithfulness via trustworthiness/continuity and neighbour preservation;
corruption geometry via centroid shift, CKA, Fréchet distance, intrinsic dimension; projected twin surfaces and margins;
always Similar-20 vs Dissimilar-20 (RQ4).

## Fast track to the paper (user, 2026-10-18: "complete this research paper as soon as possible")
One decisive Kaggle run per step; the paper is drafted alongside, so writing does not wait for the last run.

| Step | Kaggle | Output for the paper |
|---|---|---|
| A. Phase 8 gate (v5 ✅) + fixed batched solver and budget (v7) | GPU T4, v7 ~45 min | The exact method on the GPU at the cheapest precision that passes the gate; its measured speed → budget for B–C |
| B. Screening (8b) | CPU (probe, kNN) + T4 (exact Pin-GTSVM at the precision chosen in A) | Backbone comparison per corruption/severity; survivors and top 4 |
| C. Frozen main runs (10) | T4 (exact Pin-GTSVM); grid size set from the speed measured in A | RQ1, RQ2 (τ > 0 vs τ = 0), RQ4, held-out family, baselines, Demšar statistics |
| D. Fine-tuning (11), top 4 | GPU | Frozen vs fine-tuned (RQ1); recipe still to be agreed |
| E. 2D/3D analysis (12) | CPU | RQ3 figures with faithfulness measures |
| F. Selection + write-up (13–14) | — | RQ5, the paper |

The experiment framework (Phase 9) is kept minimal: one notebook per step, results appended to `results/registry.csv`.

## Next action
Step A (finish): the user runs `kaggle/notebooks/phase8_check_v7_t4.ipynb` (GPU T4, bundle v12, dataset
`phase8-references`): fixed batched solver on the slow units + GPU hours of the approved tuning protocol. Check v5
(bundle v10) passed the gate; check v6 (bundle v11) was exact but slower and hit its time limit. Check v4 (bundle v9) crashed: float32 normal equations not positive definite. Check v3 ran `phase8_check_v3_*.ipynb` on Kaggle CPU (bundle v8). Kaggle's accelerator menu offers only GPU T4 x2 and TPU v5e-8 (2026-10-18); a T4 is ~19×
slower than a P100 in float64, so the v2 plan would exceed the 12-hour session limit. Check v1 results:
`results/phase8/` (gate failed with shared landmarks; interior point chosen over FISTA). Local runs are limited to
seconds-long CPU unit tests (user rule, 2026-10-17).
