# Working rules and decisions

*Version 5 · 2026-10-16 · supersedes `12_rules_and_decisions_v4.md` (kept unchanged). Technical versions:
`../docs/rulebook.md` and `../docs/scope.md`.*

**What changed from v4:** READMEs are now single files kept up to date in place (your decision, 2026-10-16); the
numbered README copies were removed (they remain in git history).

## Your working rules (and why they protect the work)
1. **One step at a time; show and verify each step.** Mistakes are caught early, when they are cheap to fix.
2. **Don't assume — ask.** Decisions that shape the research are yours.
3. **Never change the original paper or MATLAB files.** They are the ground truth we compare against.
4. **Quality first.** Tested code, traceable numbers.
5. **Audit from time to time.** Re-check code and results with fresh eyes.
6. **Plan, then act.**
7. **Reasons must come from research and facts,** with sources.
8. **Keep planning documents in `docs/`, plain-language explanations in `Documentation/`, evidence in `results/`.**
9. **Validate every change.**
10. **When options are equally good, prefer the one that is correct in unusual cases,** and say why.
11. **Never cut** safety checks on incoming data, protection against data loss, security, accessibility, or anything you
    asked for.
12. **Ask before changing results or source files.**
13. **Use time and tokens efficiently.**
14. **No overwrites** (2026-10-13). How it works, as you decided:
    - **This `Documentation/` folder, Kaggle notebooks and code bundles:** never edited after creation; a change
      creates a new numbered file (`…_v2`, `…_v3`), and a new index lists the latest of each.
    - **READMEs:** *not* numbered — one file each, kept up to date in place; history in git (your decision, 2026-10-16).
    - **`docs/` (technical planning documents):** *not* numbered — they stay single, "living" files that are updated in
      place. Nothing is lost, because every change is saved in the local git history.
    - **Code:** keeps its file names, because two copies of the same program file can be loaded by mistake; every
      change is saved in the local git history (nothing is uploaded). The history starts with a snapshot of the whole
      project taken on 2026-10-13 (commit `c08827e`).
    - **Why this split:** nothing is ever lost, every change is recorded with what changed and why, the planning
      documents stay simple to read (one file each), and code can never get mixed up between versions.

## Project rules that follow from facts or decisions
- Split by source photo, always with a leak test (each photo exists in 75 damaged versions).
- State in every report that the classifier was trained on ImageNet-C (its authors require it).
- Label "settings chosen on the test set" results as biased.
- Check third-party data for completeness and structure before use. (The byte comparison with the official copy
  was dropped by you on 2026-10-16; reports say so.)
- Write nothing to drive C: (almost full).
- Never overwrite feature files or run results; fixed random seeds; record versions with every run.
- No passwords or tokens in any file.
- Figures readable by colour-blind readers, with labelled axes and captions.
- The fast solver is used only after matching the exact solver on all 380 pairs.
- Pin-GTSVM is described as *less sensitive* to noise, never as *removing* noise.
- Keep the system simple; every run is recorded with what changed; new combinations by settings files, not code edits;
  the selection score is fixed before the runs; size and speed reported next to accuracy.
- All computing on Kaggle; on your PC only quick tests. Rapid means less data, never fewer checks.
- Code bundles for Kaggle are built only from saved (committed) code and record which version they came from.

## Decisions so far
| Date | Decision | By | Why |
|---|---|---|---|
| 2026-10-11 | Port MATLAB to Python with corrections only | You | Test the real method, not its bugs |
| 2026-10-11 | Use Tanveer's meaning of τ | You | Matches the method the paper cites; τ = 0 and τ = 1 both work |
| 2026-10-11 | Report both setting-selection methods | You | Honest main result plus comparison with the paper |
| 2026-10-11 | CVXOPT as the maths solver | You | Closest behaviour to MATLAB on tricky cases |
| 2026-10-13 | New ImageNet-C study, not an extension of the paper | You | Your research goal |
| 2026-10-13 | Install PyTorch (CUDA 13.2) in the project environment | You (install), Claude (edition) | Edition stays supported in the next release |
| 2026-10-13 | ViTs both frozen and fine-tuned | You | Compare cheap vs adapted |
| 2026-10-13 | One-vs-one tournament | You | Balanced, affordable, recommended for kernel SVMs |
| 2026-10-13 | 20 similar + 20 different classes (instead of 100) | You | Save computing; study the effect of similarity |
| 2026-10-13 | 70/30 split by source photo | You | The paper's ratio; prevents leakage |
| 2026-10-13 | Fast landmark solver, checked against the exact one | You | Makes the setting search affordable |
| 2026-10-13 | Kaggle copy of ImageNet-C, verified against Zenodo | You | Data lives on Kaggle; must be proven correct |
| 2026-10-13 | Keep pip's cache on C: | You | Your choice |
| 2026-10-13 | Two separate 20-class experiments; class lists approved | You | Clean comparison of similar vs different |
| 2026-10-13 | New study code in `study/` | You | Keeps the port unchanged |
| 2026-10-13 | Least complexity, edge-deployable, rapid-but-thorough testing, note every change | You | Real-world use |
| 2026-10-13 | All code runs on Kaggle; quick local tests allowed; Kaggle as speed-test proxy | You | Your PC is needed for other work |
| 2026-10-13 | 7 ViTs (DINOv2 S/B, DINOv3 S/B, SigLIP 2 B, FastViT SA12/SA24) | You | See 08 |
| 2026-10-13 | Adopt held-out family test, linear + kNN baselines, two scalings, per-type reporting, statistics | You | See 09 |
| 2026-10-13 | I prepare notebooks; you run them on Kaggle | You | Your account stays with you |
| 2026-10-13 | Documentation folder in plain language | You | Understandable record of everything and why |
| 2026-10-13 | No overwrites: numbered files for documents, notebooks and bundles; git history for code | You | Full history, no mix-ups |
| 2026-10-13 | `docs/` stays as single living files (no numbered versions); history in git | You | Keep the planning documents simple |
| 2026-10-16 | Drop the byte comparison with the official Zenodo copy; close Phase 6 on the passed data checks | You | Two runs lost to download problems; the data checks had passed twice with identical results |
| 2026-10-16 | Start Phase 7 (feature extraction) | You | Move to the next step |
| 2026-10-16 | One README per folder, updated in place; numbered README copies removed | You | Simpler; history stays in git |

## Open decisions
- The single score used to pick the best combination (fixed before the runs).
- The fine-tuning recipe and budget (Phase 11).
