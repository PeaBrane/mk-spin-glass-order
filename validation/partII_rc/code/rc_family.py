"""Scale-covariant family (Theorem 1A) certificate: clean-room construction and exact check.

Heuristic part (floats, NOT part of the proof): build the zero-temperature fixed shape of
X -> |sum_{i<=n} e_i min(X_i1, X_i2)| and discretise it into an integer staircase X_c.
Rigorous part (exact integers / Fractions): G_m = CDF bound of |sum_{i<=m} e_i min(X_i1,X_i2)| via
min2 and conv from rc_core, then (C1) and (C2) of the earlier write-up checked on every
elementary interval at both endpoints using the interval's own linear pieces.
"""
import argparse
import json
import math
import time
from fractions import Fraction

import numpy as np

import rc_core as rc
from rc_core import ONE, Law


# ----------------------------------------------------------------------------- heuristic shape
def zero_t_shape(n, dx=1.0 / 1000, xmax=14.0, iters=400, tol=1e-13):
    """Float fixed point of the zero-T map normalised to E X^2 = 1.  Returns (grid, density, lam*)."""
    M = int(round(xmax / dx))
    x = np.arange(M + 1) * dx
    f = np.sqrt(2 / np.pi) * np.exp(-x * x / 2)
    lam = None
    for it in range(iters):
        # CDF by trapezoid
        F = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) * dx / 2)])
        F = np.minimum(F / F[-1], 1.0)
        fmin = 2 * f * (1 - F)
        sym = np.concatenate([fmin[:0:-1], fmin]) / 2.0  # signed density of e*min on [-xmax, xmax]
        L = len(sym)
        size = 1
        while size < n * L:
            size *= 2
        fs = np.fft.rfft(sym, size)
        conv = np.fft.irfft(fs ** n, size)[: n * (L - 1) + 1] * dx ** (n - 1)
        centre = n * (M)  # index of 0
        pos = conv[centre:]
        g = 2 * pos  # density of |S| on [0, n xmax]
        xs = np.arange(len(g)) * dx
        m0 = np.trapezoid(g, xs)
        g = g / m0
        m2 = np.trapezoid(xs * xs * g, xs)
        lam_new = math.sqrt(m2)
        fnew = lam_new * np.interp(lam_new * x, xs, g, right=0.0)
        fnew /= np.trapezoid(fnew, x)
        diff = np.max(np.abs(fnew - f))
        f = fnew
        if lam is not None and abs(lam_new - lam) < tol and diff < 1e-10:
            lam = lam_new
            break
        lam = lam_new
    return x, f, lam, it


def discretise(x, f, kappa, xmax_c, Nx, Npad):
    """Law of kappa*X conditioned on [0, xmax_c], Nx bins, padded to Npad bins; integer staircase."""
    edges = np.linspace(0.0, float(xmax_c), Nx + 1)
    # density of kappa X at y is f(y/kappa)/kappa ; CDF by fine trapezoid on the original grid
    F = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) * (x[1] - x[0]) / 2)])
    F = F / F[-1]
    Fe = np.interp(edges / kappa, x, F)
    masses = np.diff(Fe)
    masses = masses / masses.sum()
    # enforce nonincreasing (pool adjacent violators, heuristic)
    masses = np.minimum.accumulate(masses)
    masses = masses / masses.sum()
    mi = np.floor(masses * ONE).astype(np.int64)
    r = ONE - int(mi.sum())
    assert 0 <= r <= Nx
    mi[:r] += 1
    full = np.zeros(Npad, dtype=np.int64)
    full[:Nx] = mi
    law = Law(0, full)
    law.validate()
    return law


# ----------------------------------------------------------------------------- rigorous check
def G_laws(Xc, n):
    G = {1: rc.min2(Xc)}
    for m in range(2, n + 1):
        G[m] = rc.conv(G[m - 1], G[1])
    return G


class PL:
    """Exact evaluation helpers for a staircase law in index units (knot k at tau = k)."""

    def __init__(self, law):
        self.law = law
        self.C = [int(v) for v in law.cdf()]
        self.m = [int(v) for v in law.m]
        self.N = law.N

    def cdf(self, tau):
        tau = Fraction(tau)
        if tau <= 0:
            return Fraction(self.C[0], ONE) if tau == 0 else Fraction(0)
        k = tau.numerator // tau.denominator
        if k >= self.N:
            return Fraction(1)
        return Fraction(self.C[k], ONE) + Fraction(self.m[k], ONE) * (tau - k)

    def slope_on(self, ta, tb):
        """Slope (probability per index unit) on the elementary interval (ta, tb) (one bin)."""
        k = Fraction(ta).numerator // Fraction(ta).denominator
        if k >= self.N:
            return Fraction(0)
        return Fraction(self.m[k], ONE)


def check_family(Xc, hc, n, lam, A, eps0, G=None, xmax_idx=None):
    """Return dict with C1/C2 verdicts and minimal margins.  Work in index units tau = t/hc.

    X_c's knots are integers k (tau = k).  F_c(t/lam) = PLc.cdf(tau/lam).
    (C1): s(tau) = F_c(tau/lam) - G_n(tau) >= 0 on [0, lam*xmax_idx].
    (C2): s + eps0 (A/lam)(1 - F_c(tau/lam)) >= eps0 [ n g_n + sum_k C(n,k)(2A)^k eps0^{k-1} (Ht_{n-k} - G_n) ],
          g_m = slope per unit length = slope_index / hc; eps0, A in length units.
    """
    lam = Fraction(lam)
    A = Fraction(A)
    eps0 = Fraction(eps0)
    hc = Fraction(hc)
    if G is None:
        G = G_laws(Xc, n)
    PLc = PL(Xc)
    PLG = {m: PL(G[m]) for m in G}
    if xmax_idx is None:
        nz = np.nonzero(Xc.m)[0]
        xmax_idx = int(nz[-1]) + 1
    tmax = lam * xmax_idx
    knots = set()
    k = 0
    while k <= tmax:
        knots.add(Fraction(k))
        k += 1
    for kc in range(xmax_idx + 1):
        knots.add(lam * kc)
    knots = sorted(t for t in knots if t <= tmax)
    epsI = eps0 / hc  # eps0 in index units
    AI = A * hc  # A per index unit
    binom = [math.comb(n, kk) for kk in range(n + 1)]
    min_s = None
    min_s_at = None
    min_c2 = None
    min_c2_at = None
    c1_ok = True
    c2_ok = True
    nint = 0
    for ta, tb in zip(knots[:-1], knots[1:]):
        nint += 1
        # slopes of G_m on this interval (per index unit)
        sl = {m: PLG[m].slope_on(ta, tb) for m in PLG}
        for t in (ta, tb):
            Fc = PLc.cdf(t / lam)
            Gt = {m: PLG[m].cdf(t) for m in PLG}
            s = Fc - Gt[n]
            if min_s is None or s < min_s:
                min_s, min_s_at = s, t
            if s < 0:
                c1_ok = False
            H = {m: Gt[m] + n * epsI * sl[m] for m in PLG}
            lhs = s + eps0 * (A / lam) * (1 - Fc)
            rhs_inner = n * sl[n] / hc
            for kk in range(1, n + 1):
                mlow = n - kk
                if mlow == 0:
                    Ht = Fraction(1)
                else:
                    Ht = max(H[mm] for mm in range(mlow, n + 1))
                rhs_inner += binom[kk] * (2 * A) ** kk * eps0 ** (kk - 1) * (Ht - Gt[n])
            margin = lhs - eps0 * rhs_inner
            if min_c2 is None or margin < min_c2:
                min_c2, min_c2_at = margin, t
            if margin < 0:
                c2_ok = False
    return {
        "C1": c1_ok,
        "C2": c2_ok,
        "min_s": float(min_s),
        "min_s_at_t": float(min_s_at * hc),
        "min_C2_margin": float(min_c2),
        "min_C2_margin_over_eps0": float(min_c2 / eps0),
        "min_C2_at_t": float(min_c2_at * hc),
        "n_intervals": nint,
        "f_c": float(Fraction(int(Xc.m[0]), ONE) / hc),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--lam", required=True)
    ap.add_argument("--A", required=True)
    ap.add_argument("--eps0", required=True)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--xmax", default="16/5")
    ap.add_argument("--Nx", type=int, default=1600)
    ap.add_argument("--Npad", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    x, f, lamstar, its = zero_t_shape(a.n)
    xmax_c = Fraction(a.xmax)
    hc = xmax_c / a.Nx
    Npad = a.Npad or int(a.Nx * 1.6)
    Xc = discretise(x, f, a.kappa, xmax_c, a.Nx, Npad)
    G = G_laws(Xc, a.n)
    res = check_family(Xc, hc, a.n, Fraction(a.lam), Fraction(a.A), Fraction(a.eps0), G=G, xmax_idx=a.Nx)
    res.update({"n": a.n, "lam": a.lam, "A": a.A, "eps0": a.eps0, "kappa": a.kappa, "xmax": a.xmax,
                "Nx": a.Nx, "Npad": Npad, "lam_star_float": lamstar, "shape_iters": its,
                "time_s": time.time() - t0})
    np.savez_compressed(a.out + ".npz", Xc_m=Xc.m, **{f"G{m}_m": G[m].m for m in G},
                        **{f"G{m}_atom": np.int64(G[m].atom) for m in G})
    with open(a.out + ".json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
