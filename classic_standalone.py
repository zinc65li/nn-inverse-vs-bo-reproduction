"""
Standalone reproduction of the two classic (non-model / evolutionary)
baselines on the problem15 benchmark: DIRECT and CMA-ES — companions
to bo_standalone.py (GP-BO) and gen2_standalone.py (NN-inverse).

Self-contained: the only other file needed is problem15.py (the
arena). Extracted unchanged from the original hybrid-comparison
script (combo_compare.py: run_direct / run_cma + their helpers),
minus everything unrelated (no GP, no RF, no hybrids).

What the two methods are:
  * DIRECT — scipy.optimize.direct, the deterministic
    dividing-rectangles global optimizer. It evaluates the objective
    one point at a time (no batches, no model, no noise handling —
    it treats the noisy composite as if it were deterministic).
    Started after the 30-point screen with maxfun=170.
    NOTE: scipy's maxfun is a request, not a hard cap — DIRECT
    finishes its current sweep, so a run actually uses ~211-253
    total evaluations. The results therefore report BOTH the final
    value (at the actual evaluation count) and the value at exactly
    200 evaluations, read off the run's trace, so the comparison
    with the 200-capped methods stays apples-to-apples.
  * CMA-ES — pycma's CMAEvolutionStrategy, started at the screen's
    best observed point with sigma=0.25, popsize 10, 17 generations
    (= 170 evaluations, total exactly 200). Candidates are clipped
    to [0, 1] before evaluation and the clipped points are what is
    told back; fitness is the negative noisy composite (CMA-ES is
    rank-based, so only the ordering within a generation matters).
Both methods observe only the noisy scalar composite — never the
(A, B) pair — like GP-BO and inv-scalar.

Protocol (identical to the rest of the package): 30-point random
screen (counts toward the budget), seeds 300-303, scored by the
NOISELESS composite of the points chosen. Results are checkpointed
to classic_results.json after every seed.

One deliberate deviation from the original script, made so this
package is FULLY deterministic (same treatment as the other files):
the per-run observation-noise stream is seeded with
zlib.crc32(method_tag) instead of Python's process-randomized
hash(). Everything else was already deterministic (DIRECT is a
deterministic algorithm; CMA-ES takes an explicit seed). Noise draws
therefore differ from the original combo run (CMA-ES 2.312, DIRECT
2.211 there); results agree statistically. The exact finals this
package produces are listed in README.md.

Run:  python classic_standalone.py     (~1-2 minutes; needs scipy,
                                        pycma)
"""
import json
import os
import time
import warnings
import zlib

import numpy as np

warnings.filterwarnings("ignore")

import problem15 as P

D = P.D
N0, BUDGET, BATCH = 30, 200, 10
SEEDS = [300, 301, 302, 303]
CHECK = list(range(N0, BUDGET + 1, 10))


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

    def eval_one(self, x):
        return float(self.eval_batch(np.asarray(x, float).reshape(1, -1))[0])

    def arrays(self):
        return np.array(self.X), np.array(self.y)

    def curve(self):
        out, j, cur = [], 0, self.trace[0][1]
        for cp in CHECK:
            while j < len(self.trace) and self.trace[j][0] <= cp:
                cur = self.trace[j][1]; j += 1
            out.append(cur)
        return out

    def value_at(self, n):
        """Best true composite using at most n evaluations (from the
        per-evaluation trace)."""
        v = self.trace[0][1]
        for c, b in self.trace:
            if c > n:
                break
            v = b
        return float(v)

    def result(self):
        X, y = self.arrays()
        return dict(curve=self.curve(), final=self.best_true,
                    final_at_budget=self.value_at(BUDGET),
                    evals=self.count,
                    noise_chase=float(y.max() - P.true_composite(
                        X[int(np.argmax(y))])[0]))


def screen(seed):
    rng0 = np.random.default_rng(seed)
    X0 = rng0.random((N0, D))
    return X0, P.observe(X0, rng0)


# ---------------------------------------------------------------- DIRECT
def run_direct(seed, X0, y0):
    from scipy.optimize import direct
    R = Runner(seed, "DIRECT", X0, y0)
    direct(lambda x: -R.eval_one(x), [(0, 1)] * D, maxfun=BUDGET - N0)
    return R.result()


# ---------------------------------------------------------------- CMA-ES
def run_cma(seed, X0, y0):
    import cma
    R = Runner(seed, "CMA-ES", X0, y0)
    es = cma.CMAEvolutionStrategy(
        np.asarray(X0[int(np.argmax(y0))], float), 0.25,
        {"popsize": BATCH, "seed": seed, "verbose": -9})
    for _ in range((BUDGET - N0) // BATCH):
        sols = [np.clip(np.asarray(s), 0, 1) for s in es.ask()]
        newy = R.eval_batch(np.array(sols))
        es.tell(sols, list(-newy))
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
                final_at_budget=float(np.mean(
                    [p["final_at_budget"] for p in per])),
                finals_at_budget=[p["final_at_budget"] for p in per],
                evals=float(np.mean([p["evals"] for p in per])),
                noise_chase=float(np.mean([p["noise_chase"] for p in per])))


def main():
    t0 = time.time()
    out = {"opt_v": P.OPT_V, "checkpoints": CHECK, "methods": {}}
    print(f"OPT={P.OPT_V:.3f} signal={P.SIG_SD:.3f} noise={P.COMP_NOISE_SD:.3f}",
          flush=True)
    jobs = [("DIRECT", run_direct), ("CMA-ES", run_cma)]
    for name, fn in jobs:
        per = []
        for seed in SEEDS:
            X0, y0 = screen(seed)
            res = fn(seed, X0, y0)
            per.append(res)
            out["methods"][name] = summarize(per)
            with open("classic_results.json", "w") as f:
                json.dump(out, f, indent=1)
            print(f"  {name:8s} seed {seed}: final={res['final']:.3f} "
                  f"(@200: {res['final_at_budget']:.3f}) "
                  f"evals={res['evals']} ({time.time() - t0:.0f}s)", flush=True)
        s = out["methods"][name]
        print(f"[{name}] mean final = {s['final']:.3f} ±{s['final_se']:.3f} "
              f"(@200: {s['final_at_budget']:.3f}, evals~{s['evals']:.0f})",
              flush=True)

    # Combined table/plot with the BO and NN-inverse results, if present.
    combined = dict(out["methods"])
    for fn_json in ("bo_results.json", "gen2_results.json"):
        if os.path.exists(fn_json):
            with open(fn_json) as f:
                combined.update(json.load(f)["methods"])
    print("\n=========== Combined: all reference methods (this package) ===========")
    for name, ms in sorted(combined.items(), key=lambda kv: -kv[1]["final"]):
        note = (f" [@200: {ms['final_at_budget']:.3f}]"
                if "final_at_budget" in ms else "")
        print(f"{name:12s} final={ms['final']:.3f} ±{ms['final_se']:.3f}{note} "
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
    ax.set_title("All reference methods (15 parameters, 200 evaluations)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("classic_progress.png", dpi=150)
    print(f"saved classic_results.json, classic_progress.png; "
          f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
