# Phase 7 results: the 7 models, what they produced, and why

*Version 1 · 2026-10-17. Evidence: `../results/phase7/phase7_summary_v3/phase7_v3/` (with figures). Technical record:
`../docs/results.md`, section 6. Method: document 15.*

## In one paragraph
Each of the 7 chosen models looked at all 150,000 pictures and turned each one into a list of numbers (a feature
vector). Every check passed. The vectors are saved on Kaggle as the output of your notebook `phase7_features_v3`, and
every later phase reuses them, so this slow step never has to be repeated. A quick test shows all 7 models work, tells
"unrelated" classes apart almost perfectly, and finds the 20 terrier breeds much harder, which is exactly where the
models differ.

## Did we fine-tune anything? No — every model is frozen
All 7 models were used **exactly as their makers published them**. Not one number inside any model was changed:
learning was switched off, no part was retrained, and no classification layer was added. A picture goes in; the
model's own description of it comes out.
*Why frozen first:* it shows what each model already "knows" about damaged pictures, it is cheap (one pass over the
data), and the ImageNet-C authors ask that models are not trained on the damaged pictures (when we do, we must say so).
**Fine-tuning comes later (Phase 11)**, as you decided ("both frozen and fine-tuned"); its recipe is still to be agreed.
The only thing trained in Phase 7 is a tiny "sanity" classifier on top of the vectors, used purely as a check.

## Why these 7 models (from the Task 4 research, document 08)
| Model | What kind of model | Why it is in the study |
|---|---|---|
| DINOv3 S and B | Learned from 1.7 billion pictures **without any labels** | Best published robustness to damaged pictures when used frozen |
| DINOv2 S and B | The previous version, also label-free | The most used baseline; S versus B shows what model size buys |
| SigLIP 2 B | Learned by **matching pictures to their captions** | A different way of learning: does the training method matter? |
| FastViT SA12 and SA24 | Small, fast models **trained with ImageNet's labels** | Built for phones and edge devices — your edge requirement |

"S" (small, about 22 million numbers inside) and "B" (base, about 86 million) are sizes that fit Kaggle and edge budgets.

## What they produced
**Size and speed** (one picture at a time; Kaggle's T4 graphics card and 2 processor cores):

| Model | Size (million numbers) | Work per picture (billion steps) | Graphics card | Processor |
|---|---|---|---|---|
| FastViT-SA12 | 10.5 | 1.5 | 7.3 ms | **40 ms** |
| FastViT-SA24 | 20.5 | 2.9 | 13.2 ms | 76 ms |
| DINOv2-S | 21.6 | 5.5 | **6.6 ms** | 77 ms |
| DINOv3-S | 21.6 | 4.3 | 10.2 ms | 65 ms |
| DINOv2-B | 85.7 | 21.9 | 7.7 ms | 240 ms |
| DINOv3-B | 85.6 | 17.2 | 10.3 ms | 205 ms |
| SigLIP2-B | 92.9 | 17.0 | 7.2 ms | 203 ms |

**Quick accuracy check** (mildest damage level; share of test pictures named correctly):

| Model | 20 terrier breeds | 20 unrelated classes |
|---|---|---|
| FastViT-SA12 | 0.75 | 0.95 |
| FastViT-SA24 | **0.79** | 0.95 |
| DINOv2-S | 0.74 | 0.97 |
| DINOv3-S | 0.72 | 0.97 |
| DINOv2-B | **0.79** | 0.97 |
| DINOv3-B | **0.79** | **0.98** |
| SigLIP2-B | 0.67 | **0.98** |

## Why the results look like this
- **Unrelated classes are easy for everyone (95–98%).** An aircraft carrier and a cheeseburger look nothing alike, even lightly
  damaged. The differences between models should appear at stronger damage, which later phases test.
- **Terriers are hard, and that is where models differ.** The breeds look alike, so small differences in what a model
  learned decide the answer.
- **The small FastViT-SA24 ties the large models on terriers with a fraction of the work.** Part of the reason: it was
  trained with ImageNet's labels, which include these exact breeds. It never saw our test pictures (ImageNet-C is made
  from pictures kept out of training), but it was taught to tell these breeds apart — a head start the label-free DINO
  models did not get. **This must be stated whenever FastViT is compared.**
- **SigLIP 2 is last on terriers but near the top on unrelated classes.** It learned from captions, and captions say
  "a terrier", rarely which one. Its vectors capture the kind of thing in the picture well, the breed less so.
- **The summary vector beats the average of the patch vectors on terriers (by 3 to 15 points).** Averaging all patches
  mixes in background; the summary vector was trained to describe the whole object. On unrelated classes both are
  equal. The summary vector is therefore the main choice for the next phases.
- **On the graphics card, big and small models take almost the same time.** With one picture at a time the card mostly
  waits on fixed overhead per layer, not on arithmetic. The processor times show the real cost, which matters most for
  edge devices: small models about 40–77 ms, large ones 200–240 ms.
- **How sure are these numbers?** This is a quick check, not the experiment. The test pictures are 300 originals, each
  in 15 damaged versions, so differences smaller than about 3 points should not be read as real. The three-way tie at
  0.79 is a tie; SigLIP 2's 12-point gap on terriers is not noise.

## What went wrong on the way, and the fixes
| Problem | Effect | Fix |
|---|---|---|
| Speed counter crashed on SigLIP 2's pooling layer | First run stopped | Models fully frozen (no learning switched on) |
| Committed Kaggle runs never started ("Cancelled after 0 seconds") | Hours waiting, no output | Close all other GPU sessions first; check the Logs within 15 minutes |
| Kaggle kept its older timm 1.0.29 although 1.0.30 was installed | Version mismatch | Version checked before the first import. Both versions gave identical results |
| FastViT measured in its training form | Speed looked twice as slow as it is | Deployed (reparameterised) form: same vectors, correct speed |
| A stuck run could use up to 12 GPU hours | Wasted quota | Live progress lines; 30-minute limit per model; 5-minute limit on stalled picture loading |

**Repeatability:** the first (interactive) run and the final run gave the same accuracies to 4 decimals for DINOv2,
DINOv3 and SigLIP 2; FastViT differed by at most 0.16 points (expected rounding from the deployed form).

## Next
Phase 8: a fast version of the Pin-GTSVM classifier, first checked against the exact solver on all 380 class pairs.
