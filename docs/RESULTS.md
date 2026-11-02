# neuroSush: scientific results and explanations (single source of truth)

Last updated: 2026-11-02. Every number below comes from a run or file named in the "Source" column.
Rule: nothing goes on the site, in a PR or to the supervisor unless it is in this file with its caveat.

## 1. What we built (one paragraph)

neuroSush is a PyTorch library for spiking networks, HTM / Thousand Brains models and, new,
a Tolman-Eichenbaum Machine (TEM, Whittington et al. 2020 Cell). TEM separates a cognitive map into
reusable **structure g** (where am I on the graph) and **sensory identity x** (what do I see). Their
conjunction **p** is stored in a fast **Hebbian memory M** that is reset per world, while the slow weights
(transitions shared across worlds) are trained. Result: after seeing a node once, the model predicts it
when reached through a NEW route in a NEW world. Code: `src/neurosush/tem/`, teaching note: `docs/TEM.md`.

## 2. Results of our model (internal metric)

Metric (ours, not the paper's): accuracy of predicting the observation on the first visit to an already-seen
node through a **new directed edge**, in 4x4 and 8x8 graph worlds. 3 training seeds, 16,000 updates each.
Best baseline (sensory-only / frequency) is 0.1027.
Source: `experiments/tem_overnight_report.py` over `experiments/results/overnight/*.json` (in main, PR #16).

| Variant | What it is | Accuracy mean ± SD | 95% CI | Position R² (null p95) | Verdict |
|---|---|---|---|---|---|
| v0 | earlier model + reference schedules | 0.236 ± 0.150 | [-0.137, 0.608] | -0.059 (-0.045) | FAIL, unstable, max grad 3.7e6 |
| v1 | + L_p_x term, calibrated reductions, bounded attractor activation | 0.721 ± 0.038 | [0.627, 0.815] | 0.296 (-0.178) | PASS |
| v2 | v1 + reliability-weighted fusion | 0.769 ± 0.007 | [0.751, 0.787] | 0.458 (-0.045) | PASS, best learned variant |
| v3 (rotation/torus, from v0) | group structure built in | 0.832 ± 0.014 | [0.797, 0.866] | 0.264 | PASS but **structure imposed** |
| v3 (from v1) | group structure built in | 0.783 ± 0.048 | [0.664, 0.903] | 0.482 | PASS but **structure imposed** |

PASS rule: n>=3, mean >= best baseline + 0.05, every seed > frequency, CI lower bound > mean frequency.

### The four fixes that made training stable (v0 -> v1/v2)
1. Missing `L_p_x` term and mis-scaled objective: supervise sensory-cued memory, calibrate loss reductions.
2. Reliability-blind fusion -> `reliable_fusion` (weight memory correction by retrieval reliability).
3. Unbounded attractor activation -> bounded activation (prevents exploding gradients).
4. Compressed memory curriculum -> gradual memory use.
These are implementation choices tested together, not four independently proven causes (v1 groups 1+3, v2 adds 2).

## 3. The authors' public code (Spiking TEM, Kawahara & Fujisawa 2025), one run

Their TF2 code run in an isolated venv: 40k iterations, one seed, 8x8 worlds. Prediction loss stayed near
chance for 10 labels (about 2.1-2.5, ln10 = 2.30). Our v1/v2 reach cross-entropy 1.15 / 1.01 (different code and worlds, indicative only).
Spatial analysis of their checkpoint (our own scripts: `reference-code/run-spikingtem/analysis/`):

| Stream | Spatial info (null) | Units above own null p95 | Split-half r | Decoder acc (shift null p95) | Cross-world map corr | Cross-world RSA (null p95) | Distance vs grid / sensory identity |
|---|---|---|---|---|---|---|---|
| g (inferred) | 0.598 (0.174) | 97% | 0.70 | 0.163 (0.021) | -0.008 | 0.003 (0.064) | 0.018 / 0.415 |
| g_gen (prior) | 0.485 (0.312) | 91% | 0.41 | 0.038 (0.019) | -0.015 | 0.040 (0.078) | 0.101 / 0.109 |
| p (CA1) | 2.614 (0.383) | 100% | 0.89 | 0.266 (0.021) | 0.000 | 0.000 (0.040) | 0.018 / 0.959 |

Reading: strong within-world position tuning; **no evidence of shared cross-world structure**; population
distances follow sensory identity, not grid distance; gridness not measurable at 8x8; no phase machinery in
their code (only a theta term in SpikingDense); one seed. An earlier claim "g does not carry position" was
wrong (a bug: positions were not copied) and is retracted.

## 4. What we may and may not say

May say
- A clean, tested TEM implementation that trains stably at reference-scale and clearly beats baselines on our internal metric.
- Better than our earlier model (v0).
- Their public-default run stayed near chance in prediction loss; ours does not (different code/worlds, one seed).

May NOT say
- "Better than the paper" (we never scored their model on our metric; no remapping/gridness comparison).
- "Grid cells emerged" (periodicity in v3 is imposed by the rotation structure; gridness is unresolved at 8x8).
- Theta phase precession as emergent (theta is imposed).
- Any number not in section 2 or 3.

## 5. Pending science (in progress, priority 1)

| Item | Status | Owner / branch |
|---|---|---|
| Save weights, probe our models with the same metrics as section 3 (spatial info, decoder, cross-world corr/RSA, distance structure) | done for v2 s0 and v1 s0 (6.1); seeds 1-2 and 16x16 pending | `tem/spatial-probe` |
| Retrain v2 s0 and v1 s0 with checkpoints (about 2.3 h each) | completed | background |
| Live teaching plate T01 in the Atlas (Hebbian memory, new-route prediction vs baselines; g imposed, labelled) | in progress | `tem/atlas-plate` |
| Site: grid-cell chapter + results table with these caveats | waiting for site redesign | `feat/landing-neuro` |
| Real recorded-run demo of the trained model | blocked on checkpoints | after probe |

Probe results are in section 6.1.

## 6. New results (append here)

### 6.1 Spatial probe of our own trained models (2026-11-02, one seed each)

Same metric definitions as section 3, applied to our frozen checkpoints (v2 s0 and v1 s0, 16,000 updates, 8x8 worlds, new worlds with empty memory).
Source: `experiments/results/tem-spatial-v2-s0.{json,md}`, `tem-spatial-v1-s0.{json,md}`, random-init control `tem-spatial-random.{json,md}` (PR #35).

| Stream | Model | Decoder acc (shift null p95) | Cross-world map corr | Cross-world RSA (null p95) | RSA vs grid / vs sensory |
|---|---|---|---|---|---|
| g | v2 s0 | 0.246 (0.025) | 0.217 | 0.742 (0.039) | 0.009 / 0.020 |
| g | v1 s0 | 0.322 (0.024) | -0.029 | 0.833 (0.042) | -0.045 / 0.073 |
| g | random init | 0.115 (0.021) | 0.041 | 0.072 (0.161) | 0.109 / 0.161 |
| g | authors' run (section 3) | 0.163 (0.021) | -0.008 | 0.003 (0.064) | 0.018 / 0.415 |
| p | v2 s0 | 0.766 (0.030) | 0.037 | 0.066 (0.043) | 0.082 / 0.630 |
| p | v1 s0 | 0.806 (0.030) | 0.024 | 0.035 (0.043) | 0.025 / 0.730 |
| p | random init | 0.541 (0.026) | 0.022 | 0.034 (0.074) | 0.140 / 0.621 |
| p | authors' run (section 3) | 0.266 (0.021) | 0.000 | 0.000 (0.040) | 0.018 / 0.959 |

Reading:
- Position can be decoded within a world from g and from p (well above the shift null).
- **g carries a representational geometry shared across worlds**: cross-world RSA 0.74 (v2) and 0.83 (v1) against a null p95 of about 0.04, while the random-init control (0.07) stays inside its null. The authors' public run showed about 0 on this measure. Different code, worlds and training, so this is not a ranking against the paper.
- That shared geometry does **not** follow Manhattan grid distance (RSA vs grid about 0) nor sensory identity, so it is not a metric, grid-like code. It means g is a consistent, world-independent code for position; it does not show grid cells.
- p mainly follows sensory identity (RSA vs sensory 0.63 and 0.73), as in the authors' run, and its map correlation across worlds is near zero.
- Random-init p also decodes position (0.54) because the fast memory binds observations within a world: high within-world decoding alone is not evidence of learned geometry.
- Spatial information of g and g_gen is about 1e12 because the reference formula accepts signed activations: a diagnostic, not bits per spike. Only p (3.1 and 8.8e6 for v2/v1, the latter also not interpretable) has a usable-looking value for v2.
- One seed per model, 8x8 worlds: gridness is not resolvable. Needs seeds 1-2 and 16x16 worlds before any stronger statement.


## 7. Supporting facts

- Our TEM tests: `pytest tests/tem`; golden fixtures compared with measured tolerances (max float noise 1.2e-7 on outputs).
- Wall times on the M2 Pro: tests 29 s, `examples/tem_whittington.py --steps 400` 9 s, spiking example 58 s.
- Papers: Whittington et al. 2020, Cell 183, 1249-1263, doi:10.1016/j.cell.2020.10.024; Kawahara & Fujisawa 2025, bioRxiv doi:10.1101/2025.10.16.682754.
- PRs: #11 spatial tools (merged), #16 TEM models and notes (merged), site #13/#15 (merged).
