# The big picture

*Version 1 · 2026-10-13*

## Where it started
A 2025 research paper (Mehendale, Sahoo and Roul, arXiv:2501.13422) built a two-step image classifier:

1. **A pretrained neural network looks at a picture and turns it into a list of numbers** (a "feature vector") that
   describes what is in it. The paper used three classic networks: ResNet-50, AlexNet and GoogLeNet.
2. **A special classifier called Pin-GTSVM reads those numbers and decides the class.** Pin-GTSVM is a variant of the
   support vector machine that is designed to be less disturbed by noise.

Their test case was **car parking in haze**: is a parking slot busy or free when the camera image is foggy? They reported
up to 98.20 % accuracy on hazy images, compared with about 88–89 % for an earlier method that first removed the haze.

**Why this matters:** cameras in the real world see fog, rain, blur and sensor noise. A classifier that keeps working
on such images, without an extra "clean the image first" step, is simpler and cheaper to run.

## What we are doing now (your decision, 2026-10-13)
We are **not extending that paper**. We are running a **new study** that uses the same two-step idea on a much
harder and better-known test:

- **Pictures:** ImageNet-C — real photographs that were deliberately damaged in 15 ways (noise, blur, fog, snow,
  pixelation, …) at 5 strength levels.
- **Feature extractors:** modern **Vision Transformers (ViTs)** instead of the old networks.
- **Classifier:** Pin-GTSVM, rebuilt and corrected in Python.
- **Main questions:**
  1. Which ViTs give features that still work when pictures are damaged?
  2. Is Pin-GTSVM really less sensitive to damaged features than its ordinary cousin?
  3. How can we *draw* these long lists of numbers in 2D/3D so that patterns become visible, in a way that can be
     trusted and reused for similar problems?
  4. Does it matter whether the classes look alike (20 dog breeds) or completely different (20 unrelated things)?
  5. Which combination is the best one to scale up and to run on small "edge" devices such as phones?

## How the work is organised
- **One step at a time.** Each step is shown to you, checked, and approved before the next starts.
- **Everything is checked.** Code has automatic tests; data is checked against the official source; numbers come
  from saved results.
- **Heavy computing runs on Kaggle**, not on your PC.
- **Small and simple.** The final solution should be easy to put into real applications and onto small devices.

## Where we are (2026-10-13)
Done: understanding the paper and code · Python version of the code · computer setup · planning documents ·
choosing the classes · research on ViTs · research on criticisms. **In progress:** checking the ImageNet-C data on
Kaggle (Phase 6, the notebook you are running). Next: extracting features with the 7 chosen ViTs.
See [13_whats_next_v1.md](13_whats_next_v1.md).
