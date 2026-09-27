"""Clean-room exact staircase-law machinery.

Written from the text of an earlier write-up of the Part II certificate (Section 6 of the paper)
only.  No code or outputs of the main checker were consulted.

Arithmetic contract
-------------------
* Probabilities are integers in units of 2^-P (P = 48).  A law is (atom, masses) with integer
  masses m_0 >= m_1 >= ... >= 0 and atom + sum m = 2^P exactly.  It is the law of |X| for an s.u. X
  whose |X| has an atom at 0 and density m_k/h on bin [kh,(k+1)h].
* Every operation returns knot values W_k (integers) whose piecewise-linear interpolation is >= the
  exact CDF of the operation on [0, N h]; `finalize` turns them into a staircase law whose CDF is
  >= that interpolation everywhere.
* Transcendental quantities come from Arb (python-flint) balls; only ball endpoints, converted to
  exact dyadic integers with directed rounding, are used.
* Convolutions are exact integer polynomial products (FLINT fmpz_poly).
* The branch-law kernel is numba int64 code.  Every product and sum in it has an a-priori bound
  < 2^63 that is asserted before the kernel runs, so the int64 arithmetic is exact.
No floating-point value enters any inequality that is claimed rigorous.
"""
import math
from fractions import Fraction

import numba as nb
import numpy as np
from flint import arb, ctx, fmpz_poly

P = 48
ONE = 1 << P
ctx.prec = 192


# ----------------------------------------------------------------------------- arb helpers
def _dyadic(x):
    m, e = x.man_exp()
    return int(m), int(e)


def arb_ceil_scaled(x, S):
    """ceil(x * 2^S) for the upper endpoint of the ball x (so >= every point of the ball)."""
    if not x.is_finite():
        raise ValueError("non-finite arb")
    m, e = _dyadic(x.upper())
    k = e + S
    if k >= 0:
        return m << k
    return -((-m) >> (-k))


def arb_floor_scaled(x, S):
    """floor(x * 2^S) for the lower endpoint of the ball x."""
    if not x.is_finite():
        raise ValueError("non-finite arb")
    m, e = _dyadic(x.lower())
    k = e + S
    if k >= 0:
        return m << k
    return m >> (-k)


def arb_frac(fr):
    fr = Fraction(fr)
    return arb(fr.numerator) / arb(fr.denominator)


# ----------------------------------------------------------------------------- laws
class Law:
    """Staircase law on a grid of N bins; the grid spacing is kept by the caller."""

    __slots__ = ("atom", "m")

    def __init__(self, atom, m):
        self.atom = int(atom)
        self.m = np.ascontiguousarray(np.asarray(m, dtype=np.int64))

    @property
    def N(self):
        return int(self.m.shape[0])

    def cdf(self):
        c = np.empty(self.N + 1, dtype=np.int64)
        c[0] = self.atom
        np.cumsum(self.m, out=c[1:])
        c[1:] += self.atom
        return c

    def validate(self):
        assert self.atom >= 0, "negative atom"
        assert (self.m >= 0).all(), "negative mass"
        assert (np.diff(self.m) <= 0).all(), "masses not nonincreasing"
        assert self.atom + int(self.m.sum()) == ONE, "total mass != 2^P"
        return True

    def median_bin(self):
        c = self.cdf()
        return int(np.searchsorted(c, ONE // 2, side="left"))


def upper_hull(W):
    """Vertices of the least concave majorant of points (k, W[k]); exact Python ints."""
    hull = []
    for k, w in enumerate(W):
        while len(hull) >= 2:
            k1, w1 = hull[-2]
            k2, w2 = hull[-1]
            if (w2 - w1) * (k - k1) <= (w - w1) * (k2 - k1):
                hull.pop()
            else:
                break
        hull.append((k, w))
    return hull


def finalize(W):
    """Knot upper bounds W[0..N] (python ints, units 2^-P) -> staircase Law with CDF >= interp(W).

    Steps (each only raises the bound): running max; cap at 1 starting one knot before the first
    knot >= 1 (or raise the last knot to 1); least concave majorant (exact); per-segment ceil of the
    hull slope (masses nonincreasing, cumulative >= hull); trim the excess at the far end.
    """
    N = len(W) - 1
    out = [0] * (N + 1)
    cur = None
    for k in range(N + 1):
        w = int(W[k])
        cur = w if cur is None or w > cur else cur
        out[k] = cur
    first = None
    for k in range(N + 1):
        if out[k] >= ONE:
            first = k
            break
    if first is None:
        out[N] = ONE
    else:
        for k in range(max(first - 1, 0), N + 1):
            out[k] = ONE
    if out[0] < 0:
        raise ValueError("negative CDF bound at 0")
    hull = upper_hull(out)
    masses = np.zeros(N, dtype=np.int64)
    for (ka, wa), (kb, wb) in zip(hull, hull[1:]):
        num = wb - wa
        den = kb - ka
        q = -((-num) // den)
        masses[ka:kb] = q
    atom = out[0]
    csum = atom + np.cumsum(masses)
    idx = int(np.searchsorted(csum, ONE, side="left"))
    if idx < N:
        prev = atom + (int(csum[idx - 1]) - atom if idx > 0 else 0)
        masses[idx] = ONE - prev
        masses[idx + 1:] = 0
    law = Law(atom, masses)
    law.validate()
    return law


def law_cdf_at(law, h, t):
    """Exact CDF (Fraction, probability units) of a staircase law with spacing h at t >= 0."""
    t = Fraction(t)
    h = Fraction(h)
    if t < 0:
        return Fraction(0)
    q = t / h
    k = q.numerator // q.denominator
    if k >= law.N:
        return Fraction(1)
    c = law.atom + int(law.m[:k].sum())
    return Fraction(c, ONE) + Fraction(int(law.m[k]), ONE) * (q - k)


# ----------------------------------------------------------------------------- operations
def gauss_law(beta, e, N):
    """Law of beta|J|, J ~ N(0,1), on grid h = 2^-e with N bins.  CDF erf(t/(beta sqrt 2))."""
    beta = Fraction(beta)
    b = arb_frac(beta)
    h = arb(1) / arb(2) ** e
    s2 = arb(2).sqrt()
    c0 = (arb(2) / arb.pi()).sqrt() / b
    Fup = [0] * (N + 1)
    fv = [None] * (N + 2)
    for k in range(N + 2):
        t = h * k
        fv[k] = c0 * (-(t * t) / (2 * b * b)).exp()
        if k <= N:
            Fup[k] = arb_ceil_scaled((t / (b * s2)).erf(), P)
    gap = [arb_ceil_scaled(h * (fv[k] - fv[k + 1]) / 4, P) for k in range(N + 1)]
    W = [0] * (N + 1)
    W[0] = 0
    W[1] = max(arb_ceil_scaled(h * c0, P), Fup[1] + gap[1])
    for k in range(2, N + 1):
        W[k] = Fup[k] + max(gap[k - 1], gap[k])
    return finalize(W)


def coarsen(law):
    """Law on grid h -> law on grid 2h (same number of bins N, N even).

    New knot k is old knot 2k.  On new bin k the old CDF is concave piecewise linear with chord gap
    (m_{2k} - m_{2k+1})/2; new bin 0 uses the tangent at 0 (slope m_0 per old bin).
    """
    N = law.N
    assert N % 2 == 0
    C = [int(v) for v in law.cdf()]
    m = [int(v) for v in law.m] + [0, 0]
    a = law.atom
    half = N // 2
    g2 = [m[2 * k] - m[2 * k + 1] for k in range(half)] + [0]  # g2[half] = 0 (beyond old grid)
    W2 = [2 * ONE] * (N + 1)
    W2[0] = 2 * a
    for k in range(1, half + 1):
        if k == 1:
            W2[1] = max(2 * a + 4 * m[0], 2 * C[2] + g2[1])
        else:
            W2[k] = 2 * C[2 * k] + max(g2[k - 1], g2[k])
    W = [-((-w) // 2) for w in W2]
    return finalize(W)


def min2(law):
    """Law of min(|X1|,|X2|), X1, X2 iid ~ law (same grid)."""
    N = law.N
    C = [int(v) for v in law.cdf()]
    m = [int(v) for v in law.m]
    a = law.atom
    G4 = [4 * (ONE * ONE - (ONE - c) * (ONE - c)) for c in C]
    gap4 = [mm * mm for mm in m] + [0]
    W4 = [0] * (N + 1)
    W4[0] = G4[0]
    W4[1] = max(G4[0] + 8 * (ONE - a) * m[0], G4[1] + gap4[1])
    for k in range(2, N + 1):
        W4[k] = G4[k] + max(gap4[k - 1], gap4[k])
    den = 4 * ONE
    W = [-((-w) // den) for w in W4]
    return finalize(W)


def _poly_coeffs(poly, length):
    cs = [int(v) for v in poly.coeffs()]
    if len(cs) < length:
        cs.extend([0] * (length - len(cs)))
    return cs


def conv(X, Y, diag=None):
    """Law of |X + Y| for independent s.u. X, Y given by staircase laws on the same grid."""
    N = X.N
    assert Y.N == N
    p = [int(v) for v in X.m]
    q = [int(v) for v in Y.m]
    a, b = X.atom, Y.atom
    A = p[::-1] + p
    B = q[::-1] + q
    c = _poly_coeffs(fmpz_poly(A) * fmpz_poly(B), 4 * N)
    # hd(k) = h * (cont x cont density of X+Y at kh) = c[k-1+2N]/4   (units 2^-2P)
    def cc(k):
        return c[k - 1 + 2 * N]
    if cc(0) < cc(1):
        raise AssertionError("cont*cont density not nonincreasing at 0 (unimodality violated)")
    # units: 2^-2P / 16
    C16 = [16 * a * b]
    for k in range(N):
        mass16 = 16 * (a * q[k] + b * p[k]) + 4 * (cc(k) + cc(k + 1))
        C16.append(C16[-1] + mass16)
    gap16 = [abs(cc(k) - cc(k + 1)) for k in range(N)] + [0]
    W16 = [0] * (N + 1)
    W16[0] = C16[0]
    tangent16 = 16 * a * b + 16 * (a * q[0] + b * p[0]) + 8 * cc(0)
    W16[1] = max(tangent16, C16[1] + gap16[1])
    for k in range(2, N + 1):
        W16[k] = C16[k] + max(gap16[k - 1], gap16[k])
    den = 16 * ONE
    W = [-((-w) // den) for w in W16]
    if diag is not None:
        diag["conv_trunc_mass"] = 1.0 - C16[N] / (16.0 * ONE * ONE)
    return finalize(W)


def nfold(Y, n):
    """Law of |Y_1 + ... + Y_n| (iid s.u.), via a binary-doubling Lemma M chain; returns (law, ops)."""
    ops = []
    pw = {1: Y}
    k = 1
    while 2 * k <= n:
        pw[2 * k] = conv(pw[k], pw[k])
        ops.append(("conv", k, k, 2 * k))
        k *= 2
    acc = None
    acc_n = 0
    for kk in sorted(pw.keys(), reverse=True):
        if acc_n + kk <= n:
            if acc is None:
                acc, acc_n = pw[kk], kk
            else:
                acc = conv(acc, pw[kk])
                ops.append(("conv", acc_n, kk, acc_n + kk))
                acc_n += kk
    assert acc_n == n
    return acc, ops, pw


# ----------------------------------------------------------------------------- branch law
_TABLE_CACHE = {}


def delta_tables(e, s, R, m_up_max):
    """Integer tables for delta(d) = -(1/2) log(1 - e^{-2d}) at d = m*h/s, h = 2^-e.

    Dup[m] >= delta(m h/s) * 2^R / h,   Dlo[m] <= delta(m h/s) * 2^R / h  (Dlo truncated where it is 0).
    """
    key = (e, s, R, m_up_max)
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    scale = R + e
    hf = arb(1) / (arb(2) ** e * s)
    Dup = np.zeros(m_up_max + 1, dtype=np.int64)
    Dlo_list = [0]
    for mm in range(1, m_up_max + 1):
        d = hf * mm
        val = -((1 - (-(2 * d)).exp()).log()) / 2
        Dup[mm] = arb_ceil_scaled(val, scale)
    mm = 1
    while True:
        d = hf * mm
        val = -((1 - (-(2 * d)).exp()).log()) / 2
        lo = arb_floor_scaled(val, scale)
        if lo <= 0:
            break
        Dlo_list.append(lo)
        mm += 1
    Dlo = np.array(Dlo_list, dtype=np.int64)
    Dup[0] = 0
    _TABLE_CACHE[key] = (Dup, Dlo)
    return Dup, Dlo


@nb.njit(parallel=True, cache=True)
def _branch_kernel(C, m, N, s, sb, R, Omax, Dup, Dlo, i_lo, i_hi, out):
    sig = s // sb
    mask = (1 << R) - 1
    step = (1 << R) // s
    one = np.int64(1) << np.int64(48)
    m24 = (np.int64(1) << np.int64(24)) - 1
    nlo = Dlo.shape[0]
    nsb = N * sb
    for ii in nb.prange(i_hi - i_lo):
        i = i_lo + ii
        jb0 = i // sig
        base = np.int64(i) * step
        s11 = np.int64(0)
        s10 = np.int64(0)
        s01 = np.int64(0)
        s00 = np.int64(0)
        for o in range(1, Omax + 1):
            jb = jb0 + o
            if jb >= nsb:
                break
            w = m[jb // sb]
            if w == 0:
                break
            iup = jb * sig - i
            ilo = jb * sig + i
            x = base + Dup[iup]
            if ilo < nlo:
                x -= Dlo[ilo]
            bn = x >> R
            if bn >= N:
                F = one
            else:
                F = C[bn] + ((m[bn] * (x & mask) + mask) >> R)
            w1 = w >> 24
            w0 = w & m24
            F1 = F >> 24
            F0 = F & m24
            s11 += w1 * F1
            s10 += w1 * F0
            s01 += w0 * F1
            s00 += w0 * F0
        out[ii, 0] = s11
        out[ii, 1] = s10
        out[ii, 2] = s01
        out[ii, 3] = s00


def _FA_at(C, m, N, R, x):
    bn = x >> R
    if bn >= N:
        return ONE
    return int(C[bn]) + ((int(m[bn]) * (x & ((1 << R) - 1)) + (1 << R) - 1) >> R)


def branch_law(law, e, s=4, sb=4, R=14, dmax=12, diag=None, return_fine=False):
    """Law of f(A,B) = atanh(tanh A tanh B) for A, B iid ~ law on grid h = 2^-e.

    Fine points u_i = i h/s.  U_i >= P(f <= u_i) from
      P(f <= u) <= P(B <= end of the b-sub-bin containing u)
                   + sum_{later b-sub-bins} mass * F_A(x),  x >= g(u, b_left),
    with g(u,b) = atanh(tanh u / tanh b) = u + delta(b-u) - delta(b+u) exactly, g decreasing in b,
    F_A evaluated by exact interpolation at x rounded up to the grid h/2^R.  b-sub-bins beyond
    b - u > dmax use x = u + delta_up(first tail offset).  Then F(t) <= U_{i+1} on (u_i, u_{i+1}]
    (monotonicity), and the fine bounds are merged onto the coarse grid with exact chord gaps.
    """
    N = law.N
    assert s & (s - 1) == 0 and sb >= 1 and s % sb == 0 and (1 << R) % s == 0
    sig = s // sb
    h_num = Fraction(1, 2 ** e)
    Omax = int(dmax * sb * 2 ** e)
    Omax = min(Omax, N * sb, 1 << 14)
    assert Omax <= (1 << 14), "Omax too large for exact int64 accumulation"
    m_up_max = (Omax + 1) * sig + 1
    Dup, Dlo = delta_tables(e, s, R, m_up_max)
    C = law.cdf()
    m = law.m
    # a-priori overflow bounds for the int64 kernel
    assert int(m.max()) <= ONE and R <= 14
    assert (N * s + 1) * ((1 << R) // s) + int(Dup.max()) < (1 << 62)
    c_list = [int(v) for v in C]
    m_list = [int(v) for v in m]
    # fine points to compute: i = 0 .. Ns-1 while P(B <= u) < 1
    nfine = N * s
    # last i with C_B(u_i) < ONE
    last_bin = int(np.searchsorted(C, ONE, side="left"))  # first knot with C == ONE
    i_stop = min(nfine, last_bin * s + s)
    out = np.zeros((i_stop, 4), dtype=np.int64)
    if i_stop > 0:
        _branch_kernel(C, m, N, s, sb, R, Omax, Dup, Dlo, 0, i_stop, out)
    tail_off = Omax * sig + 1
    Dtail = int(Dup[tail_off])
    step = (1 << R) // s
    U = [ONE] * (nfine + 2)
    den = sb * ONE
    for i in range(i_stop):
        jb0 = i // sig
        k0b = jb0 // sb
        if k0b >= N:
            U[i] = ONE
            continue
        head = sb * c_list[k0b] + (jb0 % sb + 1) * m_list[k0b]
        S = (int(out[i, 0]) << 48) + ((int(out[i, 1]) + int(out[i, 2])) << 24) + int(out[i, 3])
        jbt = jb0 + Omax + 1
        tail = 0
        if jbt < N * sb:
            kt = jbt // sb
            TM = sb * ONE - (sb * c_list[kt] + (jbt % sb) * m_list[kt])
            if TM > 0:
                xt = i * step + Dtail
                tail = TM * _FA_at(C, m, N, R, xt)
        num = head * ONE + S + tail
        Ui = -((-num) // den)
        U[i] = min(Ui, ONE)
    # running max
    for i in range(1, nfine + 2):
        if U[i] < U[i - 1]:
            U[i] = U[i - 1]
    # coarse knots with left shift: need interp_W(u_{ks+r}) >= U_{ks+r+1}, r = 0..s-1, W_{k+1} >= U_{(k+1)s+1}
    W = [0] * (N + 1)
    gam = [0] * (N + 1)
    for k in range(N):
        base0 = U[k * s + 1]
        base1 = U[(k + 1) * s + 1]
        g = 0
        for r in range(1, s):
            v = s * U[k * s + r + 1] - (s - r) * base0 - r * base1
            if v > g:
                g = v
        gam[k] = -((-g) // s)
    for k in range(N + 1):
        gl = gam[k - 1] if k >= 1 else 0
        gr = gam[k] if k < N else 0
        W[k] = U[k * s + 1] + max(gl, gr)
    if diag is not None:
        diag["branch_U1"] = U[1] / ONE
        diag["branch_trunc_mass"] = 1.0 - U[nfine - 1] / ONE
        diag["branch_Omax"] = Omax
    res = finalize(W)
    if return_fine:
        return res, U
    return res


# ----------------------------------------------------------------------------- one RG level
def rg_step(law, e, n, s=4, sb=4, R=14, dmax=12, diag=None, keep=False):
    Y = branch_law(law, e, s=s, sb=sb, R=R, dmax=dmax, diag=diag)
    out, ops, pw = nfold(Y, n)
    if keep:
        return out, Y, ops, pw
    return out
