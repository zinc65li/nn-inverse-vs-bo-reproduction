"""
Standalone reproduction of the reference GP-BO baseline on the
problem15 benchmark — the Bayesian-optimization counterpart to
gen2_standalone.py (tandem NN-inverse variants).

Self-contained: the only other file needed is problem15.py (the arena).
Extracted unchanged from the original hybrid-comparison script
(combo_compare.py: run_gp + its helpers Runner / screen / gp_batch /
pool_for / greedy_batch), minus everything unrelated (no CMA-ES, no
DIRECT, no RF, no hybrids).

What this GP-BO is (and is not): a deliberately simple pool-based
scikit-learn implementation —
  * GaussianProcessRegressor, kernel = ConstantKernel x ARD-RBF
    (length_scale init 0.5 per dim, bounds 0.05–3.0) + WhiteKernel
    (noise init 0.3, bounds 1e-2–3.0), normalize_y=True,
    n_restarts_optimizer=0, random_state=0;
  * acquisition = expected improvement over the best OBSERVED (noisy)
    composite, evaluated on a fresh 20,000-point uniform random pool
    each round;
  * batch = greedy top-10 by EI with min pairwise distance 0.1.
It is NOT a BoTorch/Ax-grade BO (no qLogNEHVI / native multi-objective
acquisition, no gradient-based acquisition optimization). It optimizes
the scalarized composite (epsilon-augmented maximin of the two
objectives' margins) and observes only noisy composite values.

Protocol (identical to gen2_standalone.py): 30-point random screen
(counts toward the budget), then rounds of batch 10 up to 200 total
evaluations; seeds 300–303; scored by the NOISELESS composite of the
points chosen. Results are checkpointed to bo_results.json after
every seed.

One deliberate deviation from the original script, made so this
package is FULLY deterministic (same treatment as gen2_standalone.py):
the per-run observation-noise stream is seeded with
zlib.crc32(method_tag) instead of Python's process-randomized hash().
Everything else was already deterministic in the original (the screen
and the candidate pools use explicitly seeded numpy Generators; the
GP has random_state=0 and no restarts). Noise draws therefore differ
from the original combo run (GP-BO scored 2.264 there, 2.183 in the
gen_compare round); results agree statistically. The exact finals
this package produces are listed in README.md.

Fidelity note: in the original combo script the 30 screen points'
observation noise was drawn from the screen generator (screen()
below), while gen2_standalone.py's ABRunner draws screen noise from
the method's own stream. Both produce the identical X screen (same
seed) and statistically identical noise; the difference is inherited
from the original scripts and kept so each file stays a faithful
extract of the run it reproduces.

Run:  python bo_standalone.py     (~a few minutes)
"""
import json
import os
import time
import warnings
import zlib

import numpy as np
from scipy.stats import norm

warnings.filterwarnings("ignore")

import problem15 as P

D = P.D
N0, BUDGET, BATCH = 30, 200, 10
SEEDS = [300, 301, 302, 303]
CHECK = list(range(N0, BUDGET + 1, 10))
POOL_N = 20000


class Runner:
    """Evaluation harness: owns the archive of noisy composite
    observations and tracks the best NOISELESS composite (scoring
    only; the method never sees noiseless values)."""

    def __init__(self, seed, tag, X0, y0):
        self.rng = np.random.default_rng(
            seed * 7919 + zlib.crc32(tag.encode()) % 99991)
        self.X = [np.asarray(x, float) for x in X0]
        self.y = [float(v) for v in y0]
        self.count = len(X0)
        self.best_true = float(P.true_composite(np.array(self.X)).max())
        self.trace = [(self.count, self.best_true)]

    def eval_batch(self, Xnew):
        Xnew = np.atleast_2d(np.asarray(Xnew, float))
        ynew = P.observe(Xnew, self.rng)
        for x, yv in zip(Xnew, ynew):
            self.X.append(x); self.y.append(float(yv)); self.count += 1
            t = float(P.true_composite(x)[0])
            if t > self.best_true:
                self.best_true = t
            self.trace.append((self.count, self.best_true))
        return ynew

    def arrays(self):
        return np.array(self.X), np.array(self.y)

    def curve(self):
        out, j, cur = [], 0, self.trace[0][1]
        for cp in CHECK:
            while j < len(self.trace) and self.trace[j][0] <= cp:
                cur = self.trace[j][1]; j += 1
            out.append(cur)
        return out

    def result(self):
        X, y = self.arrays()
        return dict(curve=self.curve(), final=self.best_true,
                    evals=self.count,
                    noise_chase=float(y.max() - P.true_composite(
                        X[int(np.argmax(y))])[0]))


def screen(seed):
    rng0 = np.random.default_rng(seed)
    X0 = rng0.random((N0, D))
    return X0, P.observe(X0, rng0)


def greedy_batch(pool, scores, k=BATCH, min_dist=0.1):
    """Top-k by score with a minimum pairwise-distance constraint;
    fill by score order if the constraint starves the batch."""
    order = np.argsort(-scores)
    picked = []
    for idx in order:
        if all(np.linalg.norm(pool[idx] - pool[j]) >= min_dist for j in picked):
            picked.append(idx)
        if len(picked) == k:
            break
    if len(picked) < k:
        for idx in order:
            if idx not in picked:
                picked.append(idx)
            if len(picked) == k:
                break
    return pool[picked]


def gp_batch(X, y, pool, seed, rnd):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * RBF(length_scale=np.full(D, 0.5), length_scale_bounds=(0.05, 3.0))
              + WhiteKernel(noise_level=0.3, noise_level_bounds=(1e-2, 3.0)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=0, random_state=0).fit(X, y)
    mu, sd = gp.predict(pool, return_std=True)
    z = (mu - y.max()) / np.maximum(sd, 1e-9)
    ei = (mu - y.max()) * norm.cdf(z) + sd * norm.pdf(z)
    return greedy_batch(pool, ei)


def pool_for(seed, rnd):
    prng = np.random.default_rng(seed * 1000 + rnd)
    return prng.random((POOL_N, D))


def run_gp(seed, X0, y0):
    R = Runner(seed, "GP-BO", X0, y0)
    for r in range((BUDGET - N0) // BATCH):
        X, y = R.arrays()
        R.eval_batch(gp_batch(X, y, pool_for(seed, r), seed, r))
    return R.result()


# ---------------------------------------------------------------- reporting
def summarize(per):
    curves = np.array([p["curve"] for p in per])
    return dict(curve_mean=curves.mean(0).tolist(),
                curve_se=(curves.std(0, ddof=1) / np.sqrt(len(per))).tolist(),
                final=float(np.mean([p["final"] for p in per])),
                final_se=float(np.std([p["final"] for p in per], ddof=1)
                               / np.sqrt(len(per))),
                finals=[p["final"] for p in per],
                evals=float(np.mean([p["evals"] for p in per])),
                noise_chase=float(np.mean([p["noise_chase"] for p in per])))


def main():
    t0 = time.time()
    out = {"opt_v": P.OPT_V, "checkpoints": CHECK, "methods": {}}
    print(f"OPT={P.OPT_V:.3f} signal={P.SIG_SD:.3f} noise={P.COMP_NOISE_SD:.3f}",
          flush=True)
    per = []
    for seed in SEEDS:
        X0, y0 = screen(seed)
        res = run_gp(seed, X0, y0)
        per.append(res)
        out["methods"]["GP-BO"] = summarize(per)
        with open("bo_results.json", "w") as f:
            json.dump(out, f, indent=1)
        print(f"  GP-BO seed {seed}: final={res['final']:.3f} "
              f"evals={res['evals']} ({time.time() - t0:.0f}s)", flush=True)
    s = out["methods"]["GP-BO"]
    print(f"[GP-BO] mean final = {s['final']:.3f} ±{s['final_se']:.3f} "
          f"finals={[round(v, 3) for v in s['finals']]} "
          f"noise_chase={s['noise_chase']:.3f}", flush=True)

    # Combined table/plot with the NN-inverse results, if present.
    combined = {"GP-BO": s}
    if os.path.exists("gen2_results.json"):
        with open("gen2_results.json") as f:
            combined.update(json.load(f)["methods"])
        print("\n=========== Combined: GP-BO vs NN-inverse (this package) ===========")
        for name, ms in sorted(combined.items(), key=lambda kv: -kv[1]["final"]):
            print(f"{name:12s} final={ms['final']:.3f} ±{ms['final_se']:.3f} "
                  f"finals={[round(v, 2) for v in ms['finals']]}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for name, ms in combined.items():
        m, se = np.array(ms["curve_mean"]), np.array(ms["curve_se"])
        ax.plot(CHECK, m, marker="o", ms=2.5, lw=1.3, label=name)
        ax.fill_between(CHECK, m - se, m + se, alpha=0.10)
    ax.axhline(P.OPT_V, ls=":", c="k", lw=1, label="true optimum")
    ax.set_xlabel("total evaluations"); ax.set_ylabel("best TRUE composite found")
    ax.set_title("GP-BO vs NN-inverse variants (15 parameters, 200 evaluations)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("bo_progress.png", dpi=150)
    print(f"saved bo_results.json, bo_progress.png; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
