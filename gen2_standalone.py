"""
Standalone reproduction of the tandem NN-inverse variant comparison
(inv-scalar / inv-vector / inv-eps) on the problem15 benchmark.

Self-contained: the only other file needed is problem15.py (the arena).
This file contains ONLY the three inverse methods and their shared
machinery, extracted unchanged from the original experiment scripts
(gen2_compare.py / gen_compare.py / combo_compare.py), minus everything
unrelated (no GP-BO, no CMA-ES, no DIRECT, no cVAE).

Protocol: 30-point shared random screen (counts toward the budget),
then rounds of batch 10 up to 200 total evaluations; seeds 300-303;
methods are scored by the NOISELESS composite of the points they chose.
Per-seed results are checkpointed to gen2_results.json after every seed.

Two deliberate deviations from the original scripts, both made so this
package is FULLY deterministic (identical numbers on every run):
  1. The per-method observation-noise stream is seeded with
     zlib.crc32(method_tag) instead of Python's process-randomized
     hash().
  2. numpy's global RNG (used for batch sampling inside the G trainers)
     is seeded per (seed, round); the original scripts left it unseeded,
     which made even identical code vary between processes.
Noise draws therefore differ from the original reference run; results
agree statistically (ordering / spread / diagnostic bands). The exact
finals this package produces are listed in README.md — if your run does
not match them digit-for-digit, your environment differs (library
versions / CPU), and the bands are the fallback criterion.

Run:  python gen2_standalone.py     (~15 min on a 4-thread CPU)
"""
import json
import time
import warnings
import zlib

import numpy as np

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

import problem15 as P

torch.set_num_threads(4)

D = P.D
N0, BUDGET, BATCH = 30, 200, 10
SEEDS = [300, 301, 302, 303]
CHECK = list(range(N0, BUDGET + 1, 10))
EDIM = 4          # eps dimension for inv-eps
LAM_DIV = 0.1     # diversity-bonus weight for inv-eps


# ---------------------------------------------------------------- helpers
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


class MLP(nn.Module):
    def __init__(self, din, dout, sigmoid_out=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, dout))
        self.sig = sigmoid_out

    def forward(self, x):
        o = self.net(x)
        return torch.sigmoid(o) if self.sig else o


def margins(AB):
    AB = np.atleast_2d(AB)
    return np.column_stack([(AB[:, 0] - P.THR_A) / P.SCALE_A,
                            (AB[:, 1] - P.THR_B) / P.SCALE_B])


def shortboard_dir(m):
    d = np.array([1.0, 0.1]) if m[0] <= m[1] else np.array([0.1, 1.0])
    return d / np.linalg.norm(d)


def push_target(AB_point, t):
    m = margins(AB_point)[0]
    mt = m + t * shortboard_dir(m)
    return np.array([P.THR_A + mt[0] * P.SCALE_A, P.THR_B + mt[1] * P.SCALE_B])


class ABRunner:
    """Evaluation harness: owns the archive of noisy (A, B) observations
    and tracks the best NOISELESS composite (scoring only; the methods
    never see noiseless values)."""

    def __init__(self, seed, tag):
        rng0 = np.random.default_rng(seed)
        X0 = rng0.random((N0, D))
        self.rng = np.random.default_rng(
            seed * 7919 + zlib.crc32(tag.encode()) % 99991)
        self.X, self.AB, self.y = [], [], []
        self.count = 0
        self.best_true = -1e9
        self.trace = []
        self.eval_batch(X0)

    def eval_batch(self, Xnew):
        Xnew = np.atleast_2d(np.asarray(Xnew, float))
        A, B = P.objectives(Xnew)
        AB = np.column_stack([A + self.rng.normal(0, P.NOISE_SD, len(Xnew)),
                              B + self.rng.normal(0, P.NOISE_SD, len(Xnew))])
        yv = P.composite_from_AB(AB[:, 0], AB[:, 1])
        tv = P.true_composite(Xnew)
        for x, ab, y1, t1 in zip(Xnew, AB, yv, tv):
            self.X.append(x); self.AB.append(ab); self.y.append(float(y1))
            self.count += 1
            if t1 > self.best_true:
                self.best_true = float(t1)
            self.trace.append((self.count, self.best_true))
        return yv

    def arrays(self):
        return np.array(self.X), np.array(self.AB), np.array(self.y)

    def curve(self):
        out, j, cur = [], 0, self.trace[0][1]
        for cp in CHECK:
            while j < len(self.trace) and self.trace[j][0] <= cp:
                cur = self.trace[j][1]; j += 1
            out.append(cur)
        return out

    def result(self):
        y = np.array(self.y)
        return dict(curve=self.curve(), final=self.best_true, evals=self.count,
                    noise_chase=float(y.max() - P.true_composite(
                        self.X[int(np.argmax(y))])[0]))


# ---------------------------------------------------------------- training
def train_F(X, Y, seed, rnd, epochs=250):
    """Forward model F: X -> Y (scalar or vector), targets standardized."""
    torch.manual_seed(seed * 200 + rnd)
    Y = np.atleast_2d(Y.T).T if Y.ndim == 1 else Y
    ym, ys = Y.mean(0), Y.std(0) + 1e-9
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor((Y - ym) / ys, dtype=torch.float32)
    F = MLP(D, Y.shape[1])
    opt = torch.optim.Adam(F.parameters(), lr=2e-3)
    for _ in range(epochs):
        loss = ((F(xt) - yt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    F.eval()
    return F, ym, ys


def f_pred(F, ym, ys, Xc):
    with torch.no_grad():
        return F(torch.tensor(Xc, dtype=torch.float32)).numpy() * ys + ym


def train_inverse_scalar(F, ym, ys, y_arch, y_top, seed, rnd, steps=400):
    """Tandem inverse G: y -> X, trained THROUGH the frozen F.
    ym/ys are floats (scalar case)."""
    torch.manual_seed(seed * 300 + rnd)
    np.random.seed(seed * 1000 + rnd)
    G = MLP(1, D, sigmoid_out=True)
    opt = torch.optim.Adam(G.parameters(), lr=3e-3)
    pool = np.concatenate([y_arch, np.linspace(y_arch.min(), y_top, 200)])
    for _ in range(steps):
        idx = np.random.randint(0, len(pool), 256)
        yv = torch.tensor(((pool[idx] - ym) / ys).reshape(-1, 1),
                          dtype=torch.float32)
        loss = ((F(G(yv)) - yv) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    G.eval()
    return G


def diag_entry(cands, filt_scores, best_true):
    tv = P.true_composite(cands)
    return dict(top20=float(tv[np.argsort(-filt_scores)[:20]].mean()),
                mean=float(tv.mean()),
                hit=float((tv > best_true).mean()))


# ---------------------------------------------------------------- inv-scalar
def run_scalar(seed):
    R = ABRunner(seed, "inv-scalar")
    diag = []
    rnd = 0
    while R.count + BATCH <= BUDGET:
        X, AB, y = R.arrays()
        best = float(y.max()); bt = R.best_true
        F, ym, ys = train_F(X, y, seed, rnd)
        ym, ys = float(np.atleast_1d(ym)[0]), float(np.atleast_1d(ys)[0])
        G = train_inverse_scalar(F, ym, ys, y, best + 0.3, seed, rnd)
        anchors = []
        for t in np.linspace(best + 0.1, best + 0.5, 5):
            with torch.no_grad():
                anchors.append(G(torch.tensor([[(t - ym) / ys]],
                                              dtype=torch.float32)).numpy()[0])
        anchors = np.array(anchors)
        pool = np.vstack([anchors, np.clip(
            anchors[R.rng.integers(0, 5, 4000)]
            + R.rng.normal(0, 0.04, (4000, D)), 0, 1)])
        R.eval_batch(greedy_batch(pool, f_pred(F, ym, ys, pool).ravel()))
        for d in (0.3, 0.9, 1.5):  # offline diagnostics, never fed back
            with torch.no_grad():
                g = G(torch.tensor([[(best + d - ym) / ys]],
                                   dtype=torch.float32)).numpy()[0]
            dp = np.clip(g + R.rng.normal(0, 0.04, (300, D)), 0, 1)
            e = diag_entry(dp, f_pred(F, ym, ys, dp).ravel(), bt)
            e.update(delta=d, target=best + d)
            if d == 0.3:
                e["diversity"] = float(np.mean(
                    [np.linalg.norm(dp[i] - dp[j])
                     for i in range(50) for j in range(i + 1, 50)]))
            diag.append(e)
        rnd += 1
    return R.result(), diag


# ---------------------------------------------------------------- inv-vector
def train_G_vector(F, ym, ys, AB, y, seed, rnd, steps=400):
    """Tandem inverse G: (A, B) -> X, trained through the frozen F.
    Target pool: archive (A,B) + shortboard pushes of the top-5 archive
    points + jittered archive points."""
    torch.manual_seed(seed * 300 + rnd)
    np.random.seed(seed * 1000 + rnd)
    G = MLP(2, D, sigmoid_out=True)
    opt = torch.optim.Adam(G.parameters(), lr=3e-3)
    top = np.argsort(-y)[:5]
    pushes = [push_target(AB[i], t) for i in top for t in (0.1, 0.2, 0.3)]
    jit = AB[np.random.default_rng(seed).integers(0, len(AB), 100)] \
        + np.random.default_rng(seed + 1).normal(0, 1, (100, 2)) \
        * [0.1 * P.SCALE_A, 0.1 * P.SCALE_B]
    pool = np.vstack([AB, np.array(pushes), jit])
    pool_s = (pool - ym) / ys
    for _ in range(steps):
        idx = np.random.randint(0, len(pool), 256)
        abv = torch.tensor(pool_s[idx], dtype=torch.float32)
        loss = ((F(G(abv)) - abv) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    G.eval()
    return G


def g_vector(G, ym, ys, ABt):
    with torch.no_grad():
        return G(torch.tensor(((np.atleast_2d(ABt) - ym) / ys),
                              dtype=torch.float32)).numpy()


def run_vector(seed):
    R = ABRunner(seed, "inv-vector")
    diag = []
    rnd = 0
    while R.count + BATCH <= BUDGET:
        X, AB, y = R.arrays()
        bt = R.best_true
        F, ym, ys = train_F(X, AB, seed, rnd)
        G = train_G_vector(F, ym, ys, AB, y, seed, rnd)
        order = np.argsort(-y)
        anchors = [g_vector(G, ym, ys, push_target(AB[order[0]], t))[0]
                   for t in (0.1, 0.2, 0.3)]
        anchors += [g_vector(G, ym, ys, push_target(AB[i], 0.2))[0]
                    for i in order[1:4]]
        anchors = np.array(anchors)
        pool = np.vstack([anchors, np.clip(
            anchors[R.rng.integers(0, len(anchors), 4000)]
            + R.rng.normal(0, 0.04, (4000, D)), 0, 1)])
        predAB = f_pred(F, ym, ys, pool)
        R.eval_batch(greedy_batch(
            pool, P.composite_from_AB(predAB[:, 0], predAB[:, 1])))
        for t in (0.1, 0.3, 0.6):  # offline diagnostics, never fed back
            tgt = push_target(AB[order[0]], t)
            g = g_vector(G, ym, ys, tgt)[0]
            dp = np.clip(g + R.rng.normal(0, 0.04, (300, D)), 0, 1)
            pAB = f_pred(F, ym, ys, dp)
            e = diag_entry(dp, P.composite_from_AB(pAB[:, 0], pAB[:, 1]), bt)
            e.update(delta=t, target=float(P.composite_from_AB(tgt[0], tgt[1])))
            diag.append(e)
        rnd += 1
    return R.result(), diag


# ---------------------------------------------------------------- inv-eps
def train_G_eps(F, ym, ys, y, best, seed, rnd, steps=400):
    """Tandem inverse G: (y, eps) -> X. Consistency loss through frozen F
    plus a weak diversity bonus: without it, eps is ignored (ignoring eps
    also minimizes the consistency loss)."""
    torch.manual_seed(seed * 400 + rnd)
    np.random.seed(seed * 1000 + rnd)
    G = MLP(1 + EDIM, D, sigmoid_out=True)
    opt = torch.optim.Adam(G.parameters(), lr=3e-3)
    pool = np.concatenate([y, np.linspace(y.min(), best + 0.3, 200)])
    ys1 = float(np.atleast_1d(ys)[0]); ym1 = float(np.atleast_1d(ym)[0])
    for _ in range(steps):
        idx = np.random.randint(0, len(pool), 128)
        yv = ((pool[idx] - ym1) / ys1).reshape(-1, 1)
        eps = np.random.normal(size=(128, EDIM))
        inp = torch.tensor(np.hstack([yv, eps]), dtype=torch.float32)
        loss = ((F(G(inp)) - torch.tensor(yv, dtype=torch.float32)) ** 2).mean()
        y0 = float(pool[np.random.randint(0, len(pool))])
        yv0 = np.full((32, 1), (y0 - ym1) / ys1)
        inp0 = torch.tensor(np.hstack([yv0, np.random.normal(size=(32, EDIM))]),
                            dtype=torch.float32)
        out0 = G(inp0)
        loss = loss + ((F(out0) - torch.tensor(yv0, dtype=torch.float32)) ** 2).mean()
        dmat = torch.cdist(out0, out0)
        iu = torch.triu_indices(32, 32, offset=1)
        div = dmat[iu[0], iu[1]].mean()
        loss = loss - LAM_DIV * torch.clamp(div, max=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
    G.eval()
    return G


def g_eps(G, ym, ys, y_target, eps):
    ym1 = float(np.atleast_1d(ym)[0]); ys1 = float(np.atleast_1d(ys)[0])
    yv = np.full((len(eps), 1), (y_target - ym1) / ys1)
    with torch.no_grad():
        return G(torch.tensor(np.hstack([yv, eps]),
                              dtype=torch.float32)).numpy()


def run_eps(seed):
    R = ABRunner(seed, "inv-eps")
    diag = []
    rnd = 0
    while R.count + BATCH <= BUDGET:
        X, AB, y = R.arrays()
        best = float(y.max()); bt = R.best_true
        F, ym, ys = train_F(X, y, seed, rnd)
        G = train_G_eps(F, ym, ys, y, best, seed, rnd)
        pool = np.vstack([g_eps(G, ym, ys, t, R.rng.normal(size=(800, EDIM)))
                          for t in np.linspace(best + 0.1, best + 0.5, 5)])
        pool = np.clip(pool, 0, 1)
        R.eval_batch(greedy_batch(pool, f_pred(F, ym, ys, pool).ravel()))
        for d in (0.3, 0.9, 1.5):  # offline diagnostics, never fed back
            dp = np.clip(g_eps(G, ym, ys, best + d,
                               R.rng.normal(size=(300, EDIM))), 0, 1)
            e = diag_entry(dp, f_pred(F, ym, ys, dp).ravel(), bt)
            e.update(delta=d, target=best + d)
            if d == 0.3:
                s = dp[:100]
                e["diversity"] = float(np.mean(
                    [np.linalg.norm(s[i] - s[j])
                     for i in range(50) for j in range(i + 1, 50)]))
            diag.append(e)
        rnd += 1
    return R.result(), diag


# ---------------------------------------------------------------- reporting
def agg(diags):
    out = {}
    flat = [d for per in diags for d in per]
    for d in sorted({x["delta"] for x in flat}):
        sel = [x for x in flat if x["delta"] == d]
        out[str(d)] = dict(target=float(np.mean([x["target"] for x in sel])),
                           top20=float(np.mean([x["top20"] for x in sel])),
                           mean=float(np.mean([x["mean"] for x in sel])),
                           hit=float(np.mean([x["hit"] for x in sel])),
                           n=len(sel))
    divs = [x["diversity"] for x in flat if "diversity" in x]
    if divs:
        out["diversity"] = float(np.mean(divs))
    return out


def summarize(per):
    curves = np.array([p["curve"] for p in per])
    return dict(curve_mean=curves.mean(0).tolist(),
                curve_se=(curves.std(0, ddof=1) / np.sqrt(len(per))).tolist(),
                final=float(np.mean([p["final"] for p in per])),
                final_se=float(np.std([p["final"] for p in per], ddof=1)
                             / np.sqrt(len(per))),
                finals=[p["final"] for p in per],
                noise_chase=float(np.mean([p["noise_chase"] for p in per])))


def main():
    t0 = time.time()
    out = {"opt_v": P.OPT_V, "checkpoints": CHECK, "methods": {}, "diag": {}}
    jobs = [("inv-scalar", run_scalar), ("inv-vector", run_vector),
            ("inv-eps", run_eps)]
    for name, fn in jobs:
        per, diags = [], []
        for seed in SEEDS:
            res, dg = fn(seed)
            per.append(res); diags.append(dg)
            out["methods"][name] = summarize(per)
            out["diag"][name] = agg(diags)
            with open("gen2_results.json", "w") as f:
                json.dump(out, f, indent=1)
            print(f"  {name:12s} seed {seed}: final={res['final']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        print(f"[{name}] mean final = {out['methods'][name]['final']:.3f}",
              flush=True)

    print("\n=========== NN-inverse variants (15 params / 200 evals) ===========")
    for name, s in sorted(out["methods"].items(), key=lambda kv: -kv[1]["final"]):
        print(f"{name:12s} final={s['final']:.3f} ±{s['final_se']:.3f} "
              f"finals={[round(v, 2) for v in s['finals']]}")
    print("\nDiagnostics (target -> top20 / mean / hit rate):")
    for name, dg in out["diag"].items():
        for d, v in dg.items():
            if d == "diversity":
                print(f"  {name:12s} same-target diversity "
                      f"(mean pairwise distance): {v:.3f}")
            else:
                print(f"  {name:12s} step={d}: target~{v['target']:.2f} -> "
                      f"top20 {v['top20']:.2f}, mean {v['mean']:.2f}, "
                      f"hit {v['hit']:.3f} (n={v['n']})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for name, s in out["methods"].items():
        m, se = np.array(s["curve_mean"]), np.array(s["curve_se"])
        ax.plot(CHECK, m, marker="o", ms=2.5, lw=1.3, label=name)
        ax.fill_between(CHECK, m - se, m + se, alpha=0.10)
    ax.axhline(P.OPT_V, ls=":", c="k", lw=1, label="true optimum")
    ax.set_xlabel("total evaluations"); ax.set_ylabel("best TRUE composite found")
    ax.set_title("NN-inverse variants: scalar vs vector conditioning vs +eps input")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("gen2_progress.png", dpi=150)
    print(f"saved gen2_results.json, gen2_progress.png; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
