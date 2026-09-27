"""Heuristic construction of the family shape X_c (SEARCH only; floats; not part of the proof).

X_c is a discretisation of the zero-temperature fixed shape of X -> |sum_{i<=n} e_i min(X_i1, X_i2)| normalised
to E X^2 = 1, scaled by `kappa` and conditioned on [0, x_max].  Only the resulting integer staircase is used by
the certificate; how it was found is irrelevant to the proof.  (Follows validation/partII_rc/code/rc_family.py.)
"""
import math

import numpy as np

from mkcert.exact import ONE


def zero_t_shape(n, dx=1.0 / 1000, xmax=14.0, iters=400, tol=1e-13):
    """Float fixed point of the zero-T map normalised to E X^2 = 1.  Returns (grid, density, lam*, iterations)."""
    M = int(round(xmax / dx))
    x = np.arange(M + 1) * dx
    f = np.sqrt(2 / np.pi) * np.exp(-x * x / 2)
    lam = None
    it = 0
    for it in range(iters):
        F = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) * dx / 2)])
        F = np.minimum(F / F[-1], 1.0)
        fmin = 2 * f * (1 - F)
        sym = np.concatenate([fmin[:0:-1], fmin]) / 2.0
        L = len(sym)
        size = 1
        while size < n * L:
            size *= 2
        fs = np.fft.rfft(sym, size)
        conv = np.fft.irfft(fs ** n, size)[: n * (L - 1) + 1] * dx ** (n - 1)
        pos = conv[n * M:]
        g = 2 * pos
        xs = np.arange(len(g)) * dx
        g = g / np.trapezoid(g, xs)
        lam_new = math.sqrt(np.trapezoid(xs * xs * g, xs))
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
    """Integer staircase (atomless, nonincreasing masses, total 2^P) of kappa*X conditioned on [0, xmax_c]."""
    edges = np.linspace(0.0, float(xmax_c), Nx + 1)
    F = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) * (x[1] - x[0]) / 2)])
    F = F / F[-1]
    masses = np.diff(np.interp(edges / kappa, x, F))
    masses = np.minimum.accumulate(masses / masses.sum())
    masses = masses / masses.sum()
    mi = np.floor(masses * ONE).astype(np.int64)
    r = ONE - int(mi.sum())
    assert 0 <= r <= Nx
    mi[:r] += 1
    full = np.zeros(Npad, dtype=np.int64)
    full[:Nx] = mi
    return full
