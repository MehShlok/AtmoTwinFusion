# The original paper and its MATLAB code

*Version 1 · 2026-10-13*

## What the paper did
- **Goal:** decide whether a parking slot is **busy or free**, also when the image is hazy, without a separate
  haze-removal step.
- **Method:** a pretrained network (ResNet-50, AlexNet or GoogLeNet) turns each parking-slot image into a feature
  vector; a Pin-GTSVM with a curved (RBF) kernel decides busy/free.
- **Data:** PKLot (photos from two Brazilian university parking lots), CNRPark (two cameras, "A" and "B"), and a small
  custom hazy-parking set (752 original images, enlarged to 5,010 by augmentation).
- **Headline results (ResNet-50 + Pin-GTSVM):** 98.20 % on hazy images; 98.70 % training on camera B and testing on
  camera A; 89.51 % the other way round.

## What the MATLAB code contains (folder `codefilesattached/`, never modified)
| File | What it does |
|---|---|
| `Feature_Extraction (1).m` | ResNet-50 features (its last 1,000-number layer), PKLot: train UFPR05, test PUCPR |
| `Feature_Extraction_Alexnet.m` | AlexNet features (4,096 numbers), hazy-parking data |
| `FEATURE_EXTRACTION_GOOGLENET (1).m` | GoogLeNet features (1,024 numbers), CNRPark: train B, test A |
| `pinGTSVM_RBF_1 (1).m` | The Pin-GTSVM classifier itself |

## The 10 problems we found, and why each matters
1. **Settings were chosen by looking at the test results.** The code tries many settings and keeps the one with the
   best *test* accuracy. *Why it matters:* that is like grading yourself with the answer key — the reported accuracy
   is too optimistic.
2. **τ = 1 breaks the maths.** With τ = 1 the code's formula forces both lines to vanish, so in theory every image is
   called "busy". Any τ = 1 result in the paper can only come from tiny rounding leftovers of the solver. *Why it
   matters:* those numbers cannot be reproduced reliably.
3. **The straight-line (linear) option crashes.** It throws away the last feature column, so the sizes no longer fit.
   *Why it matters:* the paper's linear-vs-curved comparison must have come from other code we do not have.
4. **The maths solved is not the published Pin-GTSVM.** A lower limit is missing and one term is weighted wrongly
   compared with Tanveer et al. (2019). *Why it matters:* it is a different method than the one the paper names.
5. **Distances to the two lines are compared without scaling.** The standard rule divides by the "size" of each line.
   *Why it matters:* without it, one line can win unfairly.
6. **The "false positive" pictures shown are from the wrong model** — the last setting tried, not the best one.
7. **Results change on every run.** Splits are random without a fixed seed and not balanced by class; features are not
   scaled; one score can divide by zero.
8. **The paper and the code disagree** on some details (which GoogLeNet layer was used; two tables swap the AlexNet
   and GoogLeNet numbers; claims of GPU batching that the code does not do).
9. **ResNet-50 features were taken from its final "class score" layer**, an unusual choice; the usual layer is the one
   just before it.
10. **MATLAB's pretrained networks are not the same files as Python's.** *Why it matters:* even a perfect translation
    cannot reproduce the paper's numbers exactly.

## What this means
The paper's idea is worth testing, but its numbers cannot be used as a trustworthy baseline. That is why we
rebuilt the method in Python with these problems fixed (see [04_python_port_v1.md](04_python_port_v1.md)) and use a
fair testing procedure in the new study. Technical details: `../docs/results.md` §1.
