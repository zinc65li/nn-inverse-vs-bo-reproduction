"""
15 参数合成问题 (组合策略对比专用, 按 Xin 的实际工况: ~15 参数 / 200 次实验)。
在 problem10 的 10 参数问题上再扩展 x11..x15, 保持同一风格:
两冲突目标 + ε-maximin 复合分 + 大噪声 (每目标 sd 0.55)。
"""
import numpy as np
from scipy.optimize import minimize

D = 15
NOISE_SD = 0.55
EPS_AUG = 0.1


def objectives(X):
    X = np.atleast_2d(X)
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
     x11, x12, x13, x14, x15) = [X[:, i] for i in range(D)]
    A = (2.4 * x1 * x2 + 1.7 * np.sin(np.pi * x3) - 2.2 * (x4 - 0.7) ** 2
         + 1.1 * x5 + 0.9 * x1 * x6
         + 1.2 * np.sin(np.pi * x7) * x8 - 1.5 * (x9 - 0.35) ** 2 + 0.7 * x10 * x2
         + 1.0 * np.sin(np.pi * x11) * x12 - 1.4 * (x13 - 0.6) ** 2
         + 0.8 * x14 * x1 - 0.9 * (x15 - 0.25) ** 2)
    B = (2.1 * np.exp(-0.5 * ((x2 - 0.35) / 0.28) ** 2) + 1.5 * np.sin(np.pi * x6)
         - 1.9 * x4 ** 2 + 1.2 * x3 * x5 + 0.7 * (1 - x1) * x2
         - 1.8 * (x7 - 0.55) ** 2 + 1.1 * x8 * x9 - 1.2 * (x10 - 0.75) ** 2
         + 0.6 * np.sin(np.pi * x9)
         + 1.2 * x11 * x15 - 1.6 * (x12 - 0.45) ** 2 + 0.9 * np.sin(np.pi * x13)
         - 1.1 * (x14 - 0.3) ** 2 + 0.5 * x15)
    return A, B


_cal = np.random.default_rng(7).random((200_000, D))
_Ac, _Bc = objectives(_cal)
THR_A, THR_B = np.percentile(_Ac, 55), np.percentile(_Bc, 55)
SCALE_A, SCALE_B = _Ac.std(), _Bc.std()


def composite_from_AB(A, B):
    mA = (A - THR_A) / SCALE_A
    mB = (B - THR_B) / SCALE_B
    return np.minimum(mA, mB) + EPS_AUG * 0.5 * (mA + mB)


def true_composite(X):
    A, B = objectives(X)
    return composite_from_AB(A, B)


def observe(X, rng):
    A, B = objectives(X)
    A = A + rng.normal(0, NOISE_SD, size=len(X))
    B = B + rng.normal(0, NOISE_SD, size=len(X))
    return composite_from_AB(A, B)


_big = np.random.default_rng(11).random((800_000, D))
_bigv = true_composite(_big)
_bi = int(np.argmax(_bigv))
OPT_X, OPT_V = _big[_bi], float(_bigv[_bi])
for _x0 in _big[np.argsort(_bigv)[-25:]]:
    _r = minimize(lambda x: -true_composite(x)[0], _x0, method="Powell",
                  bounds=[(0, 1)] * D, options={"maxiter": 30000, "xtol": 1e-4, "ftol": 1e-7})
    _v = float(true_composite(np.clip(_r.x, 0, 1))[0])
    if _v > OPT_V:
        OPT_X, OPT_V = np.clip(_r.x, 0, 1), _v

SIG_SD = float(true_composite(_cal).std())
_rep = np.tile(_cal[:2000], (5, 1))
_An, _Bn = objectives(_rep)
_An = _An + np.random.default_rng(3).normal(0, NOISE_SD, _An.shape)
_Bn = _Bn + np.random.default_rng(4).normal(0, NOISE_SD, _Bn.shape)
COMP_NOISE_SD = float(composite_from_AB(_An, _Bn).reshape(5, 2000).std(axis=0).mean())

if __name__ == "__main__":
    print(f"D={D}  OPT={OPT_V:.3f} at {np.round(OPT_X, 2)}")
    print(f"signal sd={SIG_SD:.3f}  noise sd={COMP_NOISE_SD:.3f}  ratio={COMP_NOISE_SD / SIG_SD:.2f}")
    print(f"30-pt screen expected best ~ "
          f"{np.mean([true_composite(np.random.default_rng(s).random((30, D))).max() for s in range(50)]):.3f}")
