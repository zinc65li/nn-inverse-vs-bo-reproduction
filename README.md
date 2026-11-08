# Reproduction Package: GP-BO vs Tandem NN-Inverse Variants on problem15

This package supports a **four-way comparison** on one synthetic benchmark:

| | implementation |
|---|---|
| your Bayesian optimization | yours |
| your NN-inverse | yours |
| reference BO | `bo_standalone.py` (in this package) |
| reference NN-inverse | `gen2_standalone.py` (in this package) |

All four run in the same arena (`problem15.py`: 15 parameters, two
conflicting objectives combined into an ε-augmented maximin composite,
per-objective noise sd 0.55, true optimum 3.1678), under the same
protocol: 30-point random screen (counts toward the budget), then
batches of 10 up to **200 total evaluations**, seeds 300–303, scored by
the **noiseless** composite of the points each method chose. If your
implementations follow the same protocol, the numbers are directly
comparable with the reference results below.

The NN-inverse method specification is in
`nn-inverse-variants-handoff.md` (§7–§10 for the three methods, §4 for
the tandem structure design). The reference BO is specified by its
code (see "Reference BO" below).

## Contents

| file | what it is |
|---|---|
| `problem15.py` | The benchmark arena: objectives A/B, composite score, calibration constants, true optimum 3.1678. Unmodified from the original experiments. Importing it runs a ~20 s calibration; that is normal. |
| `bo_standalone.py` | The reference GP-BO baseline, extracted from the original `combo_compare.py` (`run_gp` + helpers). Writes `bo_results.json`. |
| `gen2_standalone.py` | The three tandem NN-inverse methods (inv-scalar / inv-vector / inv-eps), shared machinery, offline diagnostics, reporting. Writes `gen2_results.json`. |
| `bo_results.json` / `gen2_results.json` | Outputs of the verification runs of **this package** (see "Expected results"). Your runs should reproduce these numbers exactly. |
| `bo_progress.png` / `gen2_progress.png` | Progress plots from the same verification runs (`bo_progress.png` combines all four curves when `gen2_results.json` is present). |
| `gen2_results_statistical_check.json` | An earlier verification run of the NN package before determinism was enforced — kept as evidence that the package agrees statistically with the original experiment (see below). |

## How to run

```
python bo_standalone.py      # ~1 min;  needs scikit-learn, scipy
python gen2_standalone.py    # ~16 min on a 4-thread CPU; needs torch
```

- Results are checkpointed to the respective JSON file after every
  seed, so an interrupted run keeps completed seeds.
- Environment used for verification: Python 3.12.3, numpy 2.5.4,
  scipy 1.18.1, scikit-learn 1.9.1, torch 2.15.0+cpu, matplotlib 3.11.2.

## Expected results (deterministic — match these digit-for-digit)

The package is fully deterministic: every RNG (screen, observation
noise, torch training, batch sampling, candidate pools) is seeded.
Fresh-process re-runs of single seeds reproduced their finals to all
printed digits (GP-BO seed 301: 2.2732128384595978; inv-vector seed
301: 3.105866816072553). Expected output of full runs:

| method | mean final | SE | per-seed finals (seeds 300–303) |
|---|---|---|---|
| inv-vector | **2.713** | 0.144 | 2.487 / 3.106 / 2.754 / 2.505 |
| GP-BO (reference) | 2.235 | 0.030 | 2.158 / 2.273 / 2.216 / 2.291 |
| inv-eps | 2.183 | 0.201 | 1.670 / 2.071 / 2.572 / 2.419 |
| inv-scalar | 2.033 | 0.373 | 1.996 / 1.036 / 2.289 / 2.813 |

Diagnostics (aggregated): inv-vector generated-candidate true means
≈ 1.75 / 1.71 / 1.62 at target pushes 0.1 / 0.3 / 0.6; inv-scalar's
candidate mean falls from 0.83 (δ=0.3) to 0.10 (δ=1.5); same-target
diversity inv-eps ≈ 0.68 vs inv-scalar ≈ 0.21. GP-BO noise-chasing
(observed best − its true value) ≈ 0.65.

If your numbers differ in the third decimal or beyond, suspect library
versions or CPU/threading first (the NN code sets
`torch.set_num_threads(4)`). If they differ grossly, check the
load-bearing details in the handoff spec §11 (do-not list) —
especially: retrain F and G from scratch every round, targets
standardized with F's scalers, inv-vector's filter score = composite
of F's *predicted (A, B)*, and the diversity bonus in inv-eps.

## Reference BO (`bo_standalone.py`)

A deliberately simple pool-based scikit-learn GP — this is the exact
implementation behind the "GP-BO ≈ 2.2" figures quoted in the original
experiment reports:

- `GaussianProcessRegressor`, kernel = ConstantKernel(1.0, bounds
  1e-2–1e2) × ARD-RBF (length_scale init 0.5 per dim, bounds
  0.05–3.0) + WhiteKernel (noise init 0.3, bounds 1e-2–3.0),
  `normalize_y=True`, `n_restarts_optimizer=0`, `random_state=0`.
- Acquisition: expected improvement over the best **observed** (noisy)
  composite, evaluated on a fresh 20,000-point uniform random pool
  each round (pool seeded per (seed, round)).
- Batch: greedy top-10 by EI with min pairwise distance 0.1.
- It observes only the noisy scalar composite — never the (A, B) pair.
- It is **not** a BoTorch/Ax-grade BO: no qLogNEHVI / native
  multi-objective acquisition, no gradient-based acquisition
  optimization. Conclusions are scoped to the implementations actually
  tested; a stronger BO can legitimately do better.

Extracted from the original hybrid-comparison script with exactly one
deviation: the observation-noise stream is seeded with
`zlib.crc32("GP-BO")` instead of Python's process-randomized `hash()`
(everything else in the original was already explicitly seeded).
Fidelity note: in the original, the 30 screen points' noise is drawn
from the screen generator (`screen()` in `bo_standalone.py`), while
`gen2_standalone.py`'s runner draws screen noise from each method's
own stream. Both give the identical X screen and statistically
identical noise; the difference is inherited from the original
scripts and kept so each file stays a faithful extract.

## Relationship to the original experiments (read before comparing numbers)

Original runs reported: inv-vector 2.694 ± 0.057, inv-eps
2.233 ± 0.192, inv-scalar 1.967 ± 0.445 (GEN2 round); GP-BO 2.264
(hybrid round) and 2.183 (generative round) — the same GP code scored
differently across rounds because the original scripts seeded noise
streams with the process-randomized `hash()`. This package's
algorithms are identical, but seeding was made deterministic:

1. Observation-noise streams: `zlib.crc32(method_tag)` instead of
   Python's process-randomized `hash()`.
2. (NN only) numpy batch sampling inside the G trainers: seeded per
   (seed, round); the original left it unseeded, so even the original
   code does not reproduce its own digits between processes.

Because of this, the package's exact finals differ from the original
runs'; they agree **statistically** — same ordering
(inv-vector > GP-BO ≈ inv-eps > inv-scalar), same spread pattern
(scalar wide and bimodal, vector tight, GP tightest), same diagnostic
structure. An intermediate verification run of the NN package with
the original (unseeded) sampling is preserved in
`gen2_results_statistical_check.json` (inv-vector 2.517, inv-eps
2.467, inv-scalar 2.017) as a second statistical sample.

## Scope

- The three NN-inverse methods and the GP-BO are each standalone
  closed loops; nothing in this package combines methods. The
  DIRECT→GP-box hybrid figure (2.427) sometimes quoted alongside
  these results comes from a separate hybrid experiment and is
  **not** part of this package.
- Claims are scoped to this benchmark and these implementations:
  "inv-vector beat this GP-BO and these controls on problem15" is
  what the data shows — not a claim about BO as a category.
