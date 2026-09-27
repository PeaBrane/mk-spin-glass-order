"""Validation of rc_core against float references (NOT part of the proof)."""
import math
import time
from fractions import Fraction

import numpy as np
from scipy.special import erf

import rc_core as rc
from rc_core import ONE


def cdf_float(law, h, t):
    C = law.cdf().astype(float) / ONE
    m = law.m.astype(float) / ONE
    k = np.minimum((t / h).astype(int), law.N)
    out = np.where(k >= law.N, 1.0, C[np.minimum(k, law.N)] + m[np.minimum(k, law.N - 1)] * (t / h - k))
    return np.where(t < 0, 0.0, out)


def test_gauss():
    e, N = 6, 800
    beta = Fraction(5, 4)
    t0 = time.time()
    L = rc.gauss_law(beta, e, N)
    h = 2.0 ** -e
    t = np.linspace(0, N * h, 20001)
    ref = erf(t / (1.25 * math.sqrt(2)))
    b = cdf_float(L, h, t)
    d = b - ref
    print("gauss: min(bound-exact)=%.3e max=%.3e time %.2fs" % (d.min(), d.max(), time.time() - t0))
    return L


def mc_sum_abs(law, h, n, M=2_000_000, seed=1):
    rng = np.random.default_rng(seed)
    C = law.cdf().astype(float) / ONE
    m = law.m.astype(float) / ONE
    def sample(size):
        u = rng.random(size)
        k = np.searchsorted(C[1:], u, side="right")
        x = np.where(u <= C[0], 0.0, (k + (u - C[np.minimum(k, law.N)]) / np.maximum(m[np.minimum(k, law.N - 1)], 1e-300)) * h)
        return x
    S = np.zeros(M)
    for _ in range(n):
        S += sample(M) * rng.choice([-1, 1], M)
    return np.abs(S)


def test_conv(L, e):
    h = 2.0 ** -e
    t0 = time.time()
    W = rc.conv(L, L)
    print("conv time %.2fs" % (time.time() - t0))
    # exact float reference via fine numerical convolution of densities
    N = L.N
    # MC check instead (simpler, unbiased)
    S = mc_sum_abs(L, h, 2)
    ts = np.linspace(0, N * h, 400)
    emp = np.searchsorted(np.sort(S), ts, side="right") / len(S)
    b = cdf_float(W, h, ts)
    se = np.sqrt(np.maximum(emp * (1 - emp), 1e-12) / len(S))
    z = (emp - b) / se
    print("conv: max z (emp-bound)/se = %.2f ; max(bound-emp)=%.3e" % (z.max(), (b - emp).max()))
    return W


def branch_ref(L, h, u_arr, sub=64):
    """Float reference for P(f(A,B) <= u), A,B iid ~ L: quadrature over B with sub points per bin."""
    C = L.cdf().astype(float) / ONE
    m = L.m.astype(float) / ONE
    N = L.N
    bmid = (np.arange(N * sub) + 0.5) * h / sub
    wb = np.repeat(m / sub, sub)
    def FA(x):
        k = np.minimum((x / h).astype(np.int64), N)
        return np.where(k >= N, 1.0, C[np.minimum(k, N)] + m[np.minimum(k, N - 1)] * (x / h - k))
    out = []
    for u in u_arr:
        sel = bmid > u
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.arctanh(np.minimum(np.tanh(u) / np.tanh(bmid[sel]), 1.0))
        g = np.where(np.isfinite(g), g, 1e9)
        val = L.atom / ONE + wb[~sel].sum() + (wb[sel] * FA(g)).sum()
        out.append(val)
    return np.array(out)


def test_branch(L, e, sb=4):
    h = 2.0 ** -e
    t0 = time.time()
    Y, U = rc.branch_law(L, e, s=4, sb=sb, R=14, dmax=12, return_fine=True)
    print("branch sb=%d time %.2fs" % (sb, time.time() - t0))
    s = 4
    ii = np.arange(0, min(len(U) - 2, L.N * s), 7)
    u = ii * h / s
    ref = branch_ref(L, h, u)
    Uf = np.array([U[i] for i in ii], dtype=float) / ONE
    d = Uf - ref
    print("branch fine U - ref: min %.3e max %.3e" % (d.min(), d.max()))
    ts = np.linspace(0, L.N * h * 0.9, 300)
    ref2 = branch_ref(L, h, ts)
    b = cdf_float(Y, h, ts)
    d2 = b - ref2
    print("branch law CDF - ref: min %.3e max %.3e mean %.3e" % (d2.min(), d2.max(), d2.mean()))
    return Y


def test_coarsen(L, e):
    h = 2.0 ** -e
    Lc = rc.coarsen(L)
    ts = np.linspace(0, L.N * h, 5001)
    d = cdf_float(Lc, 2 * h, ts) - cdf_float(L, h, ts)
    print("coarsen: min %.3e max %.3e" % (d.min(), d.max()))


def test_min2(L, e):
    h = 2.0 ** -e
    M = rc.min2(L)
    ts = np.linspace(0, L.N * h, 5001)
    F = cdf_float(L, h, ts)
    ref = 1 - (1 - F) ** 2
    d = cdf_float(M, h, ts) - ref
    print("min2: min %.3e max %.3e" % (d.min(), d.max()))


if __name__ == "__main__":
    L = test_gauss()
    test_coarsen(L, 6)
    test_min2(L, 6)
    W = test_conv(L, 6)
    Y1 = test_branch(L, 6, sb=1)
    Y4 = test_branch(L, 6, sb=4)
