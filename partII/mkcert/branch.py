"""Rigorous fine-point upper bounds for the branch map f(a, b) = atanh(tanh a tanh b)  (trusted code).

Let A, B be iid with staircase law (atom, m) on the grid h = 2^-e (N bins).  For the fine points u_i = i h / s,
i = 0..Ns, this module computes integers U_i with

    U_i / 2^P >= P(f(A, B) <= u_i).

Bound.  f(a, b) <= min(a, b), so {B <= u} is contained in {f <= u}.  For b > u, f(a, b) <= u iff
a <= g(u, b) := atanh(tanh u / tanh b), and g is decreasing in b with the exact identity

    g(u, b) = u + delta(b - u) - delta(b + u),     delta(d) = -(1/2) log(1 - e^{-2d}).

Split every bin of B into sb equal sub-bins (width h/sb; B is uniform inside a bin).  Let jb0 be the sub-bin
containing u.  Then, with F_A the CDF of A and b_jb = jb h / sb the left end of sub-bin jb,

    P(f <= u) <= P(B <= b_{jb0+1}) + sum_{jb > jb0} P(B in sub-bin jb) F_A(g(u, b_jb)).

For o = jb - jb0 <= Omax the argument g(u, b_jb) is replaced by the upper bound x = u + Dup[.] - Dlo[.]
(outward-rounded delta tables, mkcert.ivdec); the sub-bins with o > Omax are bounded together by
(their total B-mass) x F_A(u + Dup[Omax*sig + 1]), valid because delta is decreasing and delta(b+u) > 0.
F_A(x) is evaluated by exact interpolation rounded up at the grid h / 2^R.  All arithmetic is on integers.

The inner sums can be evaluated by three interchangeable kernels that compute identical integers:
'python' (standard library), 'numpy' (vectorised) and 'numba' (compiled, parallel).
"""
from fractions import Fraction

from . import ivdec as V
from .exact import ONE, need

_DELTA = {}
_TABLES = {}


def delta_iv(d):
    """Interval enclosure of delta(d) = -(1/2) log(1 - e^{-2d}) for a rational d > 0 (cached by d)."""
    if d not in _DELTA:
        _DELTA[d] = V.log(1 - V.exp(V.IV.of(-2 * d))) * V.IV.of(Fraction(-1, 2))
    return _DELTA[d]


def delta_tables(e, s, R, n_up):
    """Integer tables for delta at d = k h / s, h = 2^-e, in units h / 2^R.

    Dup[k] >= delta(k h/s) 2^(R+e) for k = 1..n_up;  Dlo[k] <= delta(k h/s) 2^(R+e) for k = 1..len(Dlo)-1
    (the table stops at the first k whose lower bound is <= 0; beyond it 0 is used, valid since delta > 0).
    """
    key = (e, s, R, n_up)
    if key in _TABLES:
        return _TABLES[key]
    scale = R + e
    step = Fraction(2) ** (-e) / s
    Dup = [0] + [V.ceil_scaled(delta_iv(step * k), scale) for k in range(1, n_up + 1)]
    Dlo = [0]
    k = 1
    while True:
        v = V.floor_scaled(delta_iv(step * k), scale)
        if v <= 0:
            break
        Dlo.append(v)
        k += 1
    need(max(Dup) < (1 << 40) and max(Dlo) < (1 << 40), "delta table out of range")
    _TABLES[key] = (Dup, Dlo)
    return Dup, Dlo


# ----------------------------------------------------------------------------------------------- kernels
def _sums_python(C, m, N, s, sb, R, Omax, jb_end, Dup, Dlo, M):
    """S_i = sum_{o = 1..min(Omax, jb_end-1-jb0)} m[jb // sb] * F_A(x_{i,jb}), jb = jb0 + o, for i < M."""
    sig = s // sb
    mask = (1 << R) - 1
    step = (1 << R) // s
    nlo = len(Dlo)
    out = []
    for i in range(M):
        jb0 = i // sig
        base = i * step
        tot = 0
        for o in range(1, min(Omax, jb_end - 1 - jb0) + 1):
            jb = jb0 + o
            x = base + Dup[jb * sig - i]
            il = jb * sig + i
            if il < nlo:
                x -= Dlo[il]
            bn = x >> R
            F = ONE if bn >= N else C[bn] + ((m[bn] * (x & mask) + mask) >> R)
            tot += m[jb // sb] * F
        out.append(tot)
    return out


def _limbs_to_ints(out):
    return [(int(o[0]) << 48) + ((int(o[1]) + int(o[2])) << 24) + int(o[3]) for o in out]


def _sums_numpy(C, m, N, s, sb, R, Omax, jb_end, Dup, Dlo, M):
    """Same sums, vectorised over i with exact int64 arithmetic on 24-bit limbs."""
    import numpy as np
    C, m, Dup, Dlo = (np.asarray(v, dtype=np.int64) for v in (C, m, Dup, Dlo))
    out = np.zeros((M, 4), dtype=np.int64)
    sig = s // sb
    mask = (1 << R) - 1
    step = (1 << R) // s
    one = np.int64(ONE)
    l24 = (1 << 24) - 1
    nlo = Dlo.shape[0]
    i_all = np.arange(M, dtype=np.int64)
    jb0_all = i_all // sig
    base_all = i_all * step
    for o in range(1, Omax + 1):
        Mo = min(M, (jb_end - o) * sig)  # jb0(i) + o < jb_end  <=>  i < (jb_end - o) * sig
        if Mo <= 0:
            break
        i = i_all[:Mo]
        jb = jb0_all[:Mo] + o
        w = m[jb // sb]
        x = base_all[:Mo] + Dup[jb * sig - i]
        il = jb * sig + i
        inlo = il < nlo
        x = x - np.where(inlo, Dlo[np.where(inlo, il, 0)], 0)
        bn = x >> R
        inside = bn < N
        bi = np.where(inside, bn, 0)
        F = np.where(inside, C[bi] + ((m[bi] * (x & mask) + mask) >> R), one)
        w1, w0, F1, F0 = w >> 24, w & l24, F >> 24, F & l24
        out[:Mo, 0] += w1 * F1
        out[:Mo, 1] += w1 * F0
        out[:Mo, 2] += w0 * F1
        out[:Mo, 3] += w0 * F0
    return _limbs_to_ints(out)


_NUMBA_KERNEL = None


def _numba_kernel():
    global _NUMBA_KERNEL
    if _NUMBA_KERNEL is None:
        import numba as nb
        import numpy as np

        @nb.njit(parallel=True, cache=False)
        def kern(C, m, N, s, sb, R, Omax, jb_end, Dup, Dlo, out):
            sig = s // sb
            mask = (np.int64(1) << R) - 1
            step = (np.int64(1) << R) // s
            one = np.int64(1) << np.int64(48)
            l24 = (np.int64(1) << np.int64(24)) - 1
            nlo = Dlo.shape[0]
            for i in nb.prange(out.shape[0]):
                jb0 = i // sig
                base = np.int64(i) * step
                a11 = np.int64(0)
                a10 = np.int64(0)
                a01 = np.int64(0)
                a00 = np.int64(0)
                last = min(Omax, jb_end - 1 - jb0)
                for o in range(1, last + 1):
                    jb = jb0 + o
                    w = m[jb // sb]
                    x = base + Dup[jb * sig - i]
                    il = jb * sig + i
                    if il < nlo:
                        x -= Dlo[il]
                    bn = x >> R
                    if bn >= N:
                        F = one
                    else:
                        F = C[bn] + ((m[bn] * (x & mask) + mask) >> R)
                    w1 = w >> 24
                    w0 = w & l24
                    F1 = F >> 24
                    F0 = F & l24
                    a11 += w1 * F1
                    a10 += w1 * F0
                    a01 += w0 * F1
                    a00 += w0 * F0
                out[i, 0] = a11
                out[i, 1] = a10
                out[i, 2] = a01
                out[i, 3] = a00

        _NUMBA_KERNEL = kern
    return _NUMBA_KERNEL


def _sums_numba(C, m, N, s, sb, R, Omax, jb_end, Dup, Dlo, M):
    import numpy as np
    out = np.zeros((M, 4), dtype=np.int64)
    arr = [np.asarray(v, dtype=np.int64) for v in (C, m, Dup, Dlo)]
    _numba_kernel()(arr[0], arr[1], N, s, sb, R, Omax, jb_end, arr[2], arr[3], out)
    return _limbs_to_ints(out)


KERNELS = {"python": _sums_python, "numpy": _sums_numpy, "numba": _sums_numba}


# ----------------------------------------------------------------------------------------------- bounds
def fine_bounds(atom, m, e, s, sb, R, dmax, kernel):
    """U[0..N*s] (Python ints) with U_i / 2^P >= P(f(A,B) <= i h / s); A, B iid ~ (atom, m) on h = 2^-e."""
    m = [int(v) for v in m]
    N = len(m)
    need(s >= 1 and sb >= 1 and s % sb == 0 and (1 << R) % s == 0, "branch: bad (s, sb, R)")
    need(1 <= R <= 14, "branch: R out of range")
    need(min(m) >= 0 and max(m) <= ONE, "branch: mass out of range")
    C = [atom]
    for v in m:
        C.append(C[-1] + v)
    need(C[-1] == ONE, "branch: total mass")
    sig = s // sb
    Omax = min((int(dmax) * sb << e) if e >= 0 else (int(dmax) * sb) >> (-e), N * sb, 1 << 14)
    need(1 <= Omax <= (1 << 14), "branch: Omax out of range (int64 limb sums need Omax <= 2^14)")
    Dup, Dlo = delta_tables(e, s, R, Omax * sig + 1)
    step = (1 << R) // s
    need((N * s + 1) * step + max(Dup) < (1 << 62), "branch: x out of int64 range")
    k_end = max((k + 1 for k in range(N) if m[k] > 0), default=0)  # first bin beyond the support of B
    jb_end = k_end * sb
    M = N * s + 1
    S_all = KERNELS[kernel](C, m, N, s, sb, R, Omax, jb_end, Dup, Dlo, M)
    mask = (1 << R) - 1

    def FA_up(x):
        bn = x >> R
        return ONE if bn >= N else C[bn] + ((m[bn] * (x & mask) + mask) >> R)

    Dtail = Dup[Omax * sig + 1]
    U = [ONE] * M
    den = sb * ONE
    for i in range(M):
        jb0 = i // sig
        k0 = jb0 // sb
        if k0 >= N:
            continue
        head = sb * C[k0] + (jb0 % sb + 1) * m[k0]  # sb * 2^P * P(B <= b_{jb0+1})
        tail = 0
        jbt = jb0 + Omax + 1
        if jbt < jb_end:
            kt = jbt // sb
            TM = sb * ONE - (sb * C[kt] + (jbt % sb) * m[kt])  # sb * 2^P * P(B >= b_jbt)
            tail = TM * FA_up(i * step + Dtail)
        num = head * ONE + S_all[i] + tail
        U[i] = min(ONE, -((-num) // den))
    return U, {"Omax": Omax, "jb_end": jb_end, "n_dup": len(Dup), "n_dlo": len(Dlo)}
