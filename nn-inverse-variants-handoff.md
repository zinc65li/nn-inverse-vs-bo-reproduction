# Handoff Spec: Tandem NN-Inverse Candidate Generation — Vector-Conditioned (inv-vector) and Noise-Input (inv-eps) Variants

**Purpose.** This document specifies two variants of a tandem neural-network
inverse method for generating candidate experiments in expensive black-box
optimization, precisely enough for an independent implementation to reproduce
the benchmark results in §12. Both variants were developed and tested on
the synthetic benchmark defined in §2 under the protocol in §3. The tandem
structure itself is designed in §4. Read the whole document before coding;
§11 lists failure modes that silently invalidate results.

**Background in one paragraph.** A tandem inverse pair consists of a forward
model `F: X -> objectives` (a plain regressor trained on evaluated points)
and an inverse network `G: target -> X` trained *through the frozen F* with
the consistency loss `||F(G(target)) - target||^2`. G is never trained on
`(target, X)` pairs — direct regression of X on the target averages the
one-to-many solution set into invalid points. At generation time you feed G
a target slightly better than the best observed result, sample/perturb its
outputs into a large candidate pool, rank the pool with F, and evaluate a
diverse top batch. The loop then retrains on the enlarged archive. The two
variants differ only in what G is conditioned on:

- **inv-vector**: G is conditioned on the full objective vector `(A, B)`.
  More conditioning dimensions shrink the solution set (fiber dimension
  `D - k`), leave fewer spurious solutions that exploit F's errors, and
  empirically transform the method from a high-variance lottery into the
  strongest and most stable method on this benchmark.
- **inv-eps**: G is conditioned on a scalar target `y` plus a random noise
  vector `eps`, restoring "same target, many distinct solutions" diversity
  that a plain inverse map lacks. Requires a small diversity bonus in the
  loss (§8), otherwise `eps` is ignored.

A scalar-conditioned control (**inv-scalar**, §9) is included because the
variants are only meaningful relative to it.

---

## 1. Notation

- `D = 15` parameters, all continuous in the unit cube `[0,1]^15`.
- Archive: all evaluated points `X_i` with **noisy** objective observations
  `(A_i, B_i)` and the composite score `y_i` computed from them (§2).
- "Best" without qualification = archive point with the highest **observed**
  composite `y`. Methods never see noiseless values; noiseless ("true")
  values are used only by the benchmark harness for scoring (§3).

## 2. Benchmark problem (reproduce this arena exactly)

Two conflicting objectives over `x = (x1..x15) in [0,1]^15`
(`sin` arguments in radians):

```
A(x) = 2.4*x1*x2 + 1.7*sin(pi*x3) - 2.2*(x4-0.7)^2 + 1.1*x5 + 0.9*x1*x6
     + 1.2*sin(pi*x7)*x8 - 1.5*(x9-0.35)^2 + 0.7*x10*x2
     + 1.0*sin(pi*x11)*x12 - 1.4*(x13-0.6)^2 + 0.8*x14*x1 - 0.9*(x15-0.25)^2

B(x) = 2.1*exp(-0.5*((x2-0.35)/0.28)^2) + 1.5*sin(pi*x6) - 1.9*x4^2
     + 1.2*x3*x5 + 0.7*(1-x1)*x2
     - 1.8*(x7-0.55)^2 + 1.1*x8*x9 - 1.2*(x10-0.75)^2 + 0.6*sin(pi*x9)
     + 1.2*x11*x15 - 1.6*(x12-0.45)^2 + 0.9*sin(pi*x13) - 1.1*(x14-0.3)^2
     + 0.5*x15
```

Composite (epsilon-augmented maximin satisficing score), with margins
`mA = (A - THR_A)/SCALE_A`, `mB = (B - THR_B)/SCALE_B`:

```
composite(A, B) = min(mA, mB) + 0.1 * 0.5 * (mA + mB)
```

Calibration constants (use these exact values, or recompute as described):

| constant | value | derivation |
|---|---|---|
| THR_A | 2.9389 | 55th percentile of A over 200,000 uniform points (`numpy.random.default_rng(7)`) |
| THR_B | 3.4542 | 55th percentile of B, same sample |
| SCALE_A | 1.1543 | std of A, same sample |
| SCALE_B | 1.1609 | std of B, same sample |

Observation model: each evaluation observes `A + N(0, 0.55)` and
`B + N(0, 0.55)` (independent); the observed composite is computed from the
noisy pair. Noiseless composite std over the space ≈ 0.826; composite noise
std ≈ 0.389 (noise/signal ≈ 0.47).

True optimum of the noiseless composite: **3.1678** at approximately
`[0.989, 0.505, 0.620, 0.345, 1.000, 0.600, 0.534, 1.000, 0.546, 1.000,
0.692, 0.722, 0.516, 0.703, 1.000]`
(found by 800,000-point random search, `default_rng(11)`, plus bounded
Powell polish from the top 25 points).

## 3. Evaluation protocol

- **Budget**: 200 black-box evaluations total per run, including the screen.
- **Screen**: 30 points, `X ~ U[0,1]^15` from `numpy.random.default_rng(seed)`
  (identical X for every method at a given seed), evaluated and loaded into
  the archive before round 1.
- **Rounds**: batch size 10; run rounds while `evaluated + 10 <= 200`
  (17 rounds). Each round retrains F and G **from scratch** on the full
  archive.
- **Seeds**: 300, 301, 302, 303. Observation-noise streams are per
  (seed, method); exact stream identity is *not* part of this spec —
  4-seed means are the reproduction target (§12).
- **Scoring (harness only)**: track the best **noiseless** composite among
  all evaluated points ("final"). Also track *noise chasing* =
  `max observed y - true composite at the argmax-observed point`.
- Checkpoint results to disk after every seed; these runs are long
  (reference implementation: ~70–90 s per seed per method on CPU).

## 4. Tandem structure design

All three methods in this spec share one architecture: a forward model `F`
and an inverse network `G` chained in tandem. This section defines that
structure — topology, interfaces, gradient flow, and the reasoning behind
each structural choice. Hyperparameters are in §5; per-variant details in
§7–§9.

### 4.1 Topology and data flow

```
TRAINING (repeated every round)
  archive (X_i, objectives_i)
        |
        v
   [ F: forward MLP ]   trained by regression on the archive, then FROZEN
        ^
        |   consistency loss:  || F(G(target)) - target ||^2
        |
   [ G: inverse MLP ]   inputs are sampled TARGETS (from a target pool),
                        never archive X values

GENERATION (every round, after training)
  target(s) --> G --> anchor points --> sampling / perturbation
                                            |
                                            v
                                    candidate pool (~4000)
                                            |
                          scored by F (filter score, §5)
                                            |
                          greedy diverse top-10 batch (§5)
                                            |
                                       EVALUATE (black box)
                                            |
                          noisy observations appended to archive
                                            |
                          next round: discard F and G, retrain (§4.5)
```

The black box is only ever queried at the "EVALUATE" step. Everything else
is computation on the archive.

### 4.2 Interfaces

| variant | F: input -> output | G: input -> output |
|---|---|---|
| inv-scalar | 15 -> 1 (composite `y`) | 1 -> 15 |
| inv-vector | 15 -> 2 (`A, B`) | 2 -> 15 |
| inv-eps | 15 -> 1 (composite `y`) | (1 + 4) -> 15 (`y` concat `eps`) |

- G's final layer is a sigmoid, so every raw G output is a point of the
  search cube `[0,1]^15` by construction.
- Targets enter G **standardized with F's archive mean/std** (per output),
  and the consistency loss is computed in those standardized units (§5).
  The standardization therefore lives at the F/G interface and is part of
  the structure, not a preprocessing detail.
- `eps` (inv-eps only) is drawn fresh from `N(0, I_4)` at every use —
  training step, generation, diagnostics. It is never stored or reused.

### 4.3 Why tandem: the gradient path is the method

The inverse problem is one-to-many: the set of X mapping to one target
(the *fiber* over that target) is a manifold of dimension `D - k`
(15 - k here). Two ways to learn a map from target to X behave very
differently on that fiber:

- **Direct regression** on archive pairs `(target_i, X_i)` fits the
  conditional mean `E[X | target]` — the average over the fiber. Averaged
  candidates are typically *not* on the fiber: the mean of several good
  solutions is generally a bad solution. This is the failure the tandem
  structure exists to avoid.
- **Tandem training** updates G's parameters by the gradient of
  `||F(G(t)) - t||^2`, which flows *through* F: the term `dF/dX` evaluated
  at G's current output is F's local sensitivity map, and it tells G which
  directions in X-space move which objectives. F acts as a **differentiable
  critic**. G converges to a single self-consistent *section* of the fiber
  — one branch of solutions that F itself certifies — instead of the
  fiber's mean.

Corollary (structural weakness, measured in §10): G knows nothing about
what realistic X look like beyond what F encodes. Where F extrapolates
badly, G will find and exploit the error. The mitigations are structural,
not cosmetic: targets stay near the data (§4.5, §7–§9), the pool is
re-ranked by F but the batch is actually measured, and the loop retrains
on the enlarged archive so exploited regions get corrected by real data.

### 4.4 Training sequence (per round)

1. Fit F on the full current archive (§5).
2. Freeze F: its weights receive no updates for the rest of the round,
   while gradients still flow through it into G.
3. Build the variant's target pool (§7/§8/§9).
4. Train G **from a fresh initialization** by the consistency loss
   (inv-eps adds its diversity term, §8).
5. Generate anchor points and the candidate pool from G (§7/§8/§9).
6. Score the pool with F (filter score), select the batch greedily under
   the minimum-distance rule (§5).
7. Evaluate the batch on the black box; append the noisy observations to
   the archive. Discard F and G.

### 4.5 Structural design decisions

- **F is both teacher and filter.** The same F that trains G also ranks
  G's candidate pool at generation time. Deliberate: the pool is ranked by
  exactly the model G was optimized to satisfy, so generation and ranking
  are consistent with each other. (In benchmark runs the harness's
  noiseless scoring, §3, is what keeps the loop honest; in production use,
  re-measure any champion before reporting it — G is designed to exploit
  F, and the observed best inherits that bias.)
- **Retrain from scratch every round.** The archive grows by 10 points per
  round and F's error surface is G's curriculum; warm-starting biases G
  toward the previous round's branch. Fresh initializations also decorrelate
  successive rounds.
- **Standardization at the interface (see §4.2).** In inv-vector this is
  load-bearing: it makes the consistency loss weight A and B by their
  archive scales rather than their raw units, so neither objective
  dominates G's training by unit choice.
- **Sigmoid output head on G.** Box feasibility is structural, not a
  penalty term. Feasibility constraints beyond the box (mixtures, sums,
  categorical choices) are *not* handled by this structure and would have
  to be added as output reparameterization.
- **Diversity lives outside the networks** — in pool sampling/perturbation
  and in the min-distance greedy batch rule (§5) — except in inv-eps,
  where part of it is moved inside G via the `eps` input (§8).
- **Targets are inputs, not outputs.** Nothing in the structure chooses
  the target. Target policy (small step beyond the incumbent; shortboard
  pushes in inv-vector) is a separate, load-bearing decision; the
  diagnostics in §10 exist to measure what target choices cost.

### 4.6 What this structure does not provide

- **No uncertainty estimate** (unlike a GP surrogate). F gives point
  predictions only, so the filter cannot trade exploration against
  exploitation explicitly; exploration enters only through pool diversity
  and target placement.
- **No density or manifold model** (unlike a VAE). G does not know the
  distribution of realistic X, only F's input-output map where the target
  pool reaches.
- **No branch coverage.** A plain G returns one point per target; the
  inv-eps variant (§8) is the partial remedy.

## 5. Common machinery

**Networks.** All networks are MLPs: `Linear(in,128) -> ReLU ->
Linear(128,128) -> ReLU -> Linear(128,out)`. Inverse networks add a final
`sigmoid` so outputs lie in `[0,1]^15`. Forward networks have a linear
output head.

**Standardization.** F is trained on standardized targets (per-output mean
and std of the current archive). G consumes targets standardized with **F's
same mean/std** and its loss is computed in standardized units. X inputs are
used raw (already in `[0,1]`).

**Forward model F** (per round): full-batch Adam, lr 2e-3, 250 epochs,
MSE loss. Torch seed `seed*200 + round`.

**Tandem rule (critical).** While training G, F's weights are frozen but
gradients flow through F into G. Only G's parameters are updated. G's
training inputs are *targets*, drawn from a method-specific target pool
(§7/§8/§9), never archive X values paired with their objectives. The full
structural rationale is §4.

**Candidate filtering and batch selection.** Each round builds a pool of
~4000 candidates (method-specific, §7–§9), scores every candidate with the
filter score (method-specific), and selects a batch of 10 greedily by
descending score subject to a minimum pairwise Euclidean distance of 0.1
(in unit-cube coordinates) between selected points; if the constraint
starves the batch, fill the remaining slots by score order ignoring the
constraint. All candidates are clipped to `[0,1]^15` (G's sigmoid already
bounds its own outputs; perturbations can leave the cube).

## 6. Margins and the shortboard push (used by inv-vector)

For an objective pair `(A, B)`, margins are
`m = ((A - THR_A)/SCALE_A, (B - THR_B)/SCALE_B)`.
The **shortboard direction** pushes the lagging objective hardest:

```
d = (1.0, 0.1) if m_A <= m_B else (0.1, 1.0);  d_hat = d / ||d||
push((A,B), t):  m' = m + t * d_hat;  return (THR_A + m'_A*SCALE_A,
                                             THR_B + m'_B*SCALE_B)
```

## 7. Method A: inv-vector (vector conditioning)

Per round, with archive `(X, AB, y)`:

1. **Train F**: `X -> (A, B)` per §5 (2-output head).
2. **Train G**: `MLP(2 -> 15, sigmoid)`, Adam lr 3e-3, 400 steps,
   batch 256, torch seed `seed*300 + round`. Target pool (raw `(A,B)`
   pairs, standardized at use):
   - all archive `(A_i, B_i)`;
   - for each of the top-5 archive points by observed composite: its
     `(A,B)` pushed by `t in {0.1, 0.2, 0.3}` along *its own* shortboard
     direction (§6) — 15 points;
   - 100 jittered archive points: random archive `(A,B)` plus Gaussian
     jitter with per-objective sd `0.1 * SCALE` (i.e. ~0.115).
   Loss: mean over both outputs of `(F(G(ab)) - ab_std)^2` through frozen F.
3. **Anchors**: let `p1..p4` be the top-4 archive points by observed
   composite. Anchors are `G(push(p1, t))` for `t in {0.1, 0.2, 0.3}` plus
   `G(push(pi, 0.2))` for `i in {2,3,4}` — 6 anchors.
4. **Pool**: the 6 anchors plus 4000 perturbed copies: pick a random anchor,
   add `N(0, 0.04)` per coordinate, clip to `[0,1]`.
5. **Filter score**: `composite(F_A(x), F_B(x))` — the composite is computed
   only here (and in harness bookkeeping), never as a conditioning input.
6. Select the batch per §5, evaluate, append noisy observations to archive.

## 8. Method B: inv-eps (scalar target + noise input)

Per round, with archive `(X, AB, y)` and `best = max observed y`:

1. **Train F**: `X -> y` (scalar composite) per §5.
2. **Train G**: `MLP(1 + 4 -> 15, sigmoid)`; input is the concatenation
   `[y_std, eps]`, `eps ~ N(0, I_4)`. Adam lr 3e-3, 400 steps, torch seed
   `seed*400 + round`. Target-y pool: archive `y` values plus
   `linspace(min y, best + 0.3, 200)`. Each step computes two batches:
   - *Consistency batch*: 128 pairs `(y, eps)` with y drawn uniformly from
     the pool and fresh `eps`. Loss term: `MSE(F(G(y, eps)), y_std)`.
   - *Diversity batch*: one target `y0` drawn from the pool, 32 fresh `eps`.
     Add their consistency loss, and subtract a diversity bonus:
     `loss -= 0.1 * min(mean pairwise Euclidean distance among the 32
     outputs, 1.0)`.
   Total loss = consistency(128) + consistency(32) - bonus; backprop once
   per step. **The diversity bonus is not optional** — without it the
   consistency loss is minimized by ignoring `eps` entirely (a constant
   map is a valid optimum), and the variant collapses to a single-point
   inverse. Lambda = 0.1 and the cap of 1.0 are as tested; not tuned.
3. **Generation**: for each target in `linspace(best + 0.1, best + 0.5, 5)`,
   draw 800 fresh `eps` and evaluate G — 4000 candidates. **No output
   perturbation** in this variant; diversity must come from `eps`.
4. **Filter score**: F's predicted composite. Batch selection per §5.

## 9. Control: inv-scalar (needed for comparison)

Identical to inv-eps except: G is `MLP(1 -> 15, sigmoid)` (no `eps`
input, no diversity batch/bonus); anchors are `G(t)` for
`t in linspace(best + 0.1, best + 0.5, 5)`; the pool is the 5 anchors plus
4000 perturbed copies (random anchor + `N(0, 0.04)` per coordinate,
clipped); G's target pool and training are as in §8 step 2 without the
diversity batch (batch 256).

## 10. Diagnostics (offline; never fed back, never counted in the budget)

After each round's batch selection, using that round's F and G and the
archive state at the **start** of the round, generate 300 diagnostic
candidates per ladder step and score them with the **noiseless** composite:

- inv-scalar: ladder `delta in {0.3, 0.9, 1.5}`; candidates =
  `G(best + delta)` + 300 jittered copies (`N(0, 0.04)` per coordinate).
- inv-eps: same ladder; candidates = 300 fresh-`eps` samples at
  `best + delta`.
- inv-vector: ladder of pushes `t in {0.1, 0.3, 0.6}` applied to the best
  archive point; candidates = `G(push)` + 300 jittered copies
  (`N(0, 0.04)`).

Metrics per ladder step (aggregate by averaging over all rounds and seeds;
17 rounds x 4 seeds = 68 samples per step):

- `target`: the ladder step's target expressed as a composite value
  (for inv-vector, the composite of the pushed `(A,B)` target).
- `top20`: mean noiseless composite of the 20 candidates ranked highest by
  the method's filter score.
- `mean`: mean noiseless composite over all 300.
- `hit`: fraction of the 300 whose noiseless composite exceeds the best
  noiseless value known at round start.
- `diversity` (inv-scalar and inv-eps only, smallest ladder step only):
  mean pairwise Euclidean distance among the first 50 diagnostic
  candidates.

## 11. Failure modes / do-not list

- **Do not train G by regression on archive `(y -> X)` or `((A,B) -> X)`
  pairs.** That fits the conditional mean of a one-to-many map and produces
  averaged, invalid candidates. The tandem consistency loss through a
  frozen F is the method.
- **Do not update F while training G**, and do not co-train them jointly;
  F must be a fixed critic during G's training.
- **Do not omit the diversity bonus in inv-eps** (see §8). Check the
  diversity diagnostic (§10): if it is near the inv-scalar jitter level
  (~0.2), `eps` is being ignored and the run is invalid.
- **Do not condition on the composite in inv-vector.** Conditioning inputs
  are the raw objectives; the composite appears only in the filter score,
  target selection (top-k by composite), and bookkeeping.
- **Do not let diagnostic evaluations enter the archive** or count against
  the budget; they exist only to measure target-vs-achievement.
- **Do not reuse F or G across rounds.** Retrain from scratch each round on
  the full archive; stale networks measurably degrade the loop.
- Targets must stay near the data (scalar: at most `best + 0.5` in the
  generation ladder; vector: pushes `t <= 0.3` for generation anchors).
  Large extrapolations are for diagnostics only — realized candidate
  quality collapses there (§12), which is a result to reproduce, not a
  bug to fix.
- Report per-seed finals, not just the mean: inv-scalar is bimodal, and a
  mean-only report hides its defining behavior.

## 12. Reference results (the run this spec describes)

Final best noiseless composite after 200 evaluations, 4 seeds:

| method | mean | SE | per-seed finals | noise chasing |
|---|---|---|---|---|
| **inv-vector** | **2.694** | 0.057 | 2.614 / 2.637 / 2.663 / 2.864 | 0.652 |
| inv-eps | 2.233 | 0.192 | 2.095 / 1.752 / 2.557 / 2.526 | 1.049 |
| inv-scalar | 1.967 | 0.445 | 2.545 / 1.275 / 2.907 / 1.140 | 0.994 |

Diagnostics (aggregated over 68 round-samples per step):

| method | ladder step | target | top20 | mean | hit |
|---|---|---|---|---|---|
| inv-scalar | delta 0.3 | 2.27 | 0.53 | 0.55 | 0.134 |
| inv-scalar | delta 0.9 | 2.87 | 0.16 | 0.16 | 0.063 |
| inv-scalar | delta 1.5 | 3.47 | -0.15 | -0.18 | 0.048 |
| inv-vector | t 0.1 | 2.55 | 1.55 | 1.49 | 0.176 |
| inv-vector | t 0.3 | 2.71 | 1.51 | 1.45 | 0.173 |
| inv-vector | t 0.6 | 2.86 | 1.43 | 1.37 | 0.145 |
| inv-eps | delta 0.3 | 2.63 | 1.04 | 0.72 | 0.085 |
| inv-eps | delta 0.9 | 3.23 | 0.75 | 0.49 | 0.033 |
| inv-eps | delta 1.5 | 3.83 | 0.44 | 0.24 | 0.010 |

Diversity diagnostic: inv-eps **0.482** vs inv-scalar **0.206**.

Context from companion experiments on the *same* arena and protocol
(separate runs, different noise streams — order-of-magnitude references
only): GP-based Bayesian optimization 2.183 ± 0.037; DIRECT-then-GP hybrid
2.427; CMA-ES 2.312; cVAE closed loop 1.431 (target +0.3) and 1.181
(target +1.5). inv-vector's 2.694 is the highest mean of any method tested
on this arena.

## 13. Acceptance criteria for a reproduction

1. Ordering of 4-seed mean finals: **inv-vector > inv-eps > inv-scalar**,
   with inv-vector's mean ≳ 2.5.
2. inv-scalar shows large seed spread (range ≳ 1.0 across the 4 seeds);
   inv-vector's spread is small (range ≲ 0.4).
3. Diagnostics: inv-vector candidate `mean` at t=0.1 in the 1.4–1.6 band;
   inv-scalar `mean` at delta=1.5 at or below ~0; inv-eps diversity
   clearly above inv-scalar's (~0.4–0.5 vs ~0.2).
4. Sanity checks before the full run: on the 30-point screen alone, F's
   in-sample fit should be near-perfect (it will overfit — expected);
   G's round-1 anchors must lie in `[0,1]^15`; the first diagnostic batch
   must not enter the archive (archive size grows by exactly 10 per round).

## 14. Reference implementation map

The original implementation (Python 3.12, PyTorch 2.15 CPU, numpy,
scikit-learn not required for these two variants) lives in:

- `problem15.py` — benchmark problem, calibration, optimum.
- `gen2_compare.py` — inv-scalar / inv-vector / inv-eps runners,
  diagnostics, aggregation, plotting (`gen2_results.json`,
  `gen2_progress.png`).
- `gen_compare.py` — shared MLP definition and the scalar tandem trainer
  reused by the control.
- `GEN2_RESULTS.md` — the results report this spec's §12 is drawn from.

## 15. Scope notes and open variants

- This benchmark has **two** objectives. The production setting that
  motivated the work has **three** (A, B, C): the vector variant extends
  trivially (F gains an output, G an input, the shortboard rule pushes the
  minimum-margin objective with 0.1 on the other two), and the fiber
  argument (`D - k`) predicts the advantage should hold or grow. That
  extension is **not** benchmarked in the reference run.
- The shortboard push direction is a design choice of the reference run.
  Alternatives (front-normal push, proportional margin scaling) were not
  compared.
- inv-vector + `eps` (vector conditioning *and* a noise input) is the
  obvious combination and was **not** tested in the reference run.
- The diversity-bonus weight (0.1) and `eps` dimension (4) were set a
  priori and not tuned.
