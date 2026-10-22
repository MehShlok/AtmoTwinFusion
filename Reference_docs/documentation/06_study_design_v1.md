# The study design, decision by decision

*Version 1 · 2026-10-13. Each decision says what was decided, by whom, and why.*

## The data: ImageNet-C
- **What it is:** 50,000 real photographs (ImageNet's validation set, 50 per class, 1,000 classes), each damaged in
  15 ways — noise (3 kinds), blur (4), weather (snow, frost, fog, brightness) and digital (contrast, elastic warping,
  pixelation, JPEG compression) — at 5 strengths. So every photo exists in **75 damaged versions**
  (Hendrycks and Dietterich, 2019).
- **Its rule, and how we handle it:** its authors say models should be trained on clean ImageNet and only *tested* on
  ImageNet-C; training on it is "discouraged" and must be stated. You decided to train and test on ImageNet-C, so
  **every report says so**, and our numbers are **not compared with published ImageNet-C leaderboards**.
- **Where it comes from:** a 67.8 GB copy on Kaggle (`vthutrang/imagenet-c`). *Why we check it:* it was uploaded by an
  unknown person, so before using it we compare it with the official copy (Zenodo, about 66 GB) — Phase 6.

## Classes: 20 similar + 20 completely different (your decision)
Two **separate** 20-class experiments: 20 terrier breeds that look very alike, and 20 classes with nothing in common.
*Why 20, not 100:* to keep computing time affordable. *Why two kinds:* to see whether damage hurts more when classes
are hard to tell apart. Details: [07_choosing_the_classes_v1.md](07_choosing_the_classes_v1.md).
That gives 40 classes × 75 versions × 50 photos = **150,000 pictures**.

## Train/test split: 70/30 **by source photo** (confirmed by you)
For each class, 35 of the 50 photos (with all their 75 versions) go to training and 15 to testing.
*Why by source photo:* if the foggy version of a photo is in training and its blurry version is in testing, the test is
like an exam whose questions were already seen with a different font — the score would be falsely high. The source
paper itself saw this "memorising" effect (its Section 4.3.1). *Why 70/30:* the paper's own ratio.
Inside training, the photos are further split into **5 cross-validation groups**, again by source photo.

## ViTs: frozen and fine-tuned (your decision)
Both ways are tested. *Why:* frozen is cheap and easy to deploy; fine-tuning shows how much adapting the model to
damaged pictures helps. The 7 chosen ViTs: [08_choosing_the_vits_v1.md](08_choosing_the_vits_v1.md).

## Classifier: Pin-GTSVM, one-vs-one (your decision)
190 pair-contests per class set (380 in total). *Why one-vs-one:* see [02_key_ideas_explained_v1.md](02_key_ideas_explained_v1.md).

## Fast "landmark" solver (approved by you)
The exact method takes 6–11 s per pair, so trying many settings on 380 pairs would take days. The landmark method is
far faster. *Safety rule:* it is used only after it gives the same answers as the exact method on all 380 pairs.

## Two ways of choosing settings (your decision: report both)
- **Main:** settings chosen by cross-validation on training photos only; test photos used once at the end.
- **For comparison only:** the paper's way (settings chosen on the test photos), always labelled "optimistically biased".

## Comparison methods (approved by you after the Task 5 research)
Next to Pin-GTSVM we run: the ordinary twin SVM (τ = 0), a standard SVM, a **linear classifier** and **nearest
neighbours (kNN)**. *Why:* if a simple straight-line classifier does just as well, the simpler one wins — it is
smaller and faster on small devices.

## Extra tests (approved by you)
- **Held-out damage family:** train on three damage families (e.g. noise, blur, weather) and test on the fourth
  (e.g. digital), for all four rotations. *Why:* research shows models trained on some kinds of damage often fail on
  new kinds (Geirhos et al., 2018); this tells us how the method copes with damage it has never seen.
- **Two ways of scaling the feature numbers** (L2 normalisation vs standardisation). *Why:* curved (RBF) kernels are
  sensitive to the scale of the numbers.
- **Results per damage type and per family**, plus each ViT's training augmentations and data-leak status. *Why:*
  some ViTs were trained with blur-like augmentations, which can flatter their blur results (Mintun et al., 2021).
- **Proper statistics** (Demšar, 2006). *Why:* with hundreds of comparisons some differences appear by pure chance.

## Your working principles
- **Rapid but thorough:** time is saved by using a *subset* of the data, never by skipping checks.
- **Note every change:** every run is recorded with what changed compared with the previous one.
- **Least complexity, deployable:** few dependencies; the final model needs only the ViT plus a small set of numbers
  for the classifier; size and speed are reported next to accuracy.
- **Pick the best, then scale up:** after the tests, the best combination is tried on more data.
- **The best combination is judged by a score fixed before the runs.** *Why:* choosing the score after seeing the
  results would let us pick whichever score flatters our method.
