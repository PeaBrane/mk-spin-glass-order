"""Construction of the staircase laws (NOT trusted: every output is re-verified by the checks).

Standard library only.  The constructors follow validation/partII_rc/code/rc_core.py.  Each builds
integer knot values W_k whose piecewise-linear interpolation is meant to lie above the exact CDF of the
operation, and `finalize` turns W into a staircase law (running maximum, cap at 1, least concave majorant,
rounding up).  The checker re-verifies every output against the exact operation (conv, min2, coarsen), against
the rigorous fine bounds (branch) or against interval enclosures (Gaussian start), so a bug here can only make
a certificate fail, never make a false one pass.  The one piece of this module whose output is not a law,
`nfold_plan`, is verified separately (exact.check_nfold_plan).
"""
from fractions import Fraction

from . import ivdec as V
from .exact import ONE, P, need, polymul


class Law:
    """Staircase law: atom and masses (lists of Python ints) in units 2^-P on a grid of N bins."""

    __slots__ = ("atom", "m")

    def __init__(self, atom, m):
        self.atom = int(atom)
        self.m = [int(v) for v in m]

    @property
    def N(self):
        return len(self.m)

    def cdf(self):
        C = [self.atom]
        for v in self.m:
            C.append(C[-1] + v)
        return C

    def pair(self):
        return self.atom, self.m

    def median_bin(self):
        c = self.atom
        if c >= ONE // 2:
            return 0
        for k, v in enumerate(self.m):
            c += v
            if c >= ONE // 2:
                return k + 1
        return self.N


def upper_hull(W):
    hull = []
    for k, w in enumerate(W):
        while len(hull) >= 2:
            (k1, w1), (k2, w2) = hull[-2], hull[-1]
            if (w2 - w1) * (k - k1) <= (w - w1) * (k2 - k1):
                hull.pop()
            else:
                break
        hull.append((k, w))
    return hull


def finalize(W):
    """Knot values W[0..N] (Python ints, units 2^-P) -> staircase Law whose CDF is >= interp(W) everywhere."""
    N = len(W) - 1
    out = []
    cur = None
    for w in W:
        w = int(w)
        cur = w if cur is None or w > cur else cur
        out.append(cur)
    first = next((k for k in range(N + 1) if out[k] >= ONE), None)
    if first is None:
        out[N] = ONE
    else:
        for k in range(max(first - 1, 0), N + 1):
            out[k] = ONE
    need(out[0] >= 0, "finalize: negative value at 0")
    masses = [0] * N
    hull = upper_hull(out)
    for (ka, wa), (kb, wb) in zip(hull, hull[1:]):
        q = -((wa - wb) // (kb - ka))  # ceil of the hull slope
        for k in range(ka, kb):
            masses[k] = q
    atom = out[0]
    c = atom
    for k in range(N):
        if c + masses[k] >= ONE:  # trim: the CDF reaches 1 in bin k
            masses[k] = ONE - c
            for kk in range(k + 1, N):
                masses[kk] = 0
            break
        c += masses[k]
    need(atom + sum(masses) == ONE and all(masses[k] >= masses[k + 1] for k in range(N - 1)) and masses[-1] >= 0,
         "finalize produced an invalid law")
    return Law(atom, masses)


# ----------------------------------------------------------------------------------------------- operations
def gauss_law(beta, e, N):
    """Law of beta |J|, J ~ N(0,1), on the grid h = 2^-e with N bins (CDF erf(t / (beta sqrt 2)))."""
    beta = Fraction(beta)
    h = Fraction(2) ** (-e)
    bs2 = V.IV.of(beta) * V.sqrt(V.IV.of(2))  # beta sqrt 2
    c0 = V.sqrt(V.IV.of(2) / V.pi()) / beta  # density at 0
    Fup = [V.ceil_scaled(V.erf(V.IV.of(h * k) / bs2), P) for k in range(N + 1)]
    fv = [c0 * V.exp(-V.IV.of(h * h * k * k / (2 * beta * beta))) for k in range(N + 2)]
    gap = [V.ceil_scaled((fv[k] - fv[k + 1]) * h / 4, P) for k in range(N + 1)]
    W = [0] * (N + 1)
    W[1] = max(V.ceil_scaled(c0 * h, P), Fup[1] + gap[1])
    for k in range(2, N + 1):
        W[k] = Fup[k] + max(gap[k - 1], gap[k])
    return finalize(W)


def coarsen(law):
    """Grid h -> 2h, same number of bins N (N even): new knot k = old knot 2k plus the exact chord gap."""
    N = law.N
    need(N % 2 == 0, "coarsen: N odd")
    C = law.cdf()
    m = law.m + [0, 0]
    half = N // 2
    g2 = [m[2 * k] - m[2 * k + 1] for k in range(half)] + [0]
    W2 = [2 * ONE] * (N + 1)
    W2[0] = 2 * law.atom
    for k in range(1, half + 1):
        if k == 1:
            W2[1] = max(2 * law.atom + 4 * m[0], 2 * C[2] + g2[1])
        else:
            W2[k] = 2 * C[2 * k] + max(g2[k - 1], g2[k])
    return finalize([-((-w) // 2) for w in W2])


def min2(law):
    """Law of min(|X1|, |X2|), X1, X2 iid ~ law."""
    N = law.N
    C = law.cdf()
    m = law.m
    a = law.atom
    G4 = [4 * (ONE * ONE - (ONE - c) * (ONE - c)) for c in C]
    gap4 = [mm * mm for mm in m] + [0]
    W4 = [0] * (N + 1)
    W4[0] = G4[0]
    W4[1] = max(G4[0] + 8 * (ONE - a) * m[0], G4[1] + gap4[1])
    for k in range(2, N + 1):
        W4[k] = G4[k] + max(gap4[k - 1], gap4[k])
    return finalize([-((-w) // (4 * ONE)) for w in W4])


def conv(X, Y):
    """Law of |X + Y| for independent symmetric X, Y with staircase laws on the same grid."""
    N = X.N
    need(Y.N == N, "conv: grid mismatch")
    p, q = X.m, Y.m
    a, b = X.atom, Y.atom
    cs = polymul(p[::-1] + p, q[::-1] + q)  # signed-grid product
    cs += [0] * (4 * N - len(cs))

    def cc(k):  # 4 h * (cont x cont density of X+Y at kh), units 2^-2P
        return cs[k - 1 + 2 * N]

    need(cc(0) >= cc(1), "conv: density not nonincreasing at 0")
    C16 = [16 * a * b]
    for k in range(N):
        C16.append(C16[-1] + 16 * (a * q[k] + b * p[k]) + 4 * (cc(k) + cc(k + 1)))
    gap16 = [abs(cc(k) - cc(k + 1)) for k in range(N)] + [0]
    W16 = [0] * (N + 1)
    W16[0] = C16[0]
    W16[1] = max(16 * a * b + 16 * (a * q[0] + b * p[0]) + 8 * cc(0), C16[1] + gap16[1])
    for k in range(2, N + 1):
        W16[k] = C16[k] + max(gap16[k - 1], gap16[k])
    return finalize([-((-w) // (16 * ONE)) for w in W16])


def branch_from_bounds(U, N, s):
    """Staircase law Y with CDF_Y(u_i) >= U_{i+1} for all fine points (merge onto the coarse grid)."""
    U = list(U)
    for i in range(1, len(U)):
        if U[i] < U[i - 1]:
            U[i] = U[i - 1]
    U = U + [ONE]  # U[N*s + 1]
    gam = [0] * N
    for k in range(N):
        base0, base1 = U[k * s + 1], U[(k + 1) * s + 1]
        g = 0
        for r in range(1, s):
            g = max(g, s * U[k * s + r + 1] - (s - r) * base0 - r * base1)
        gam[k] = -((-g) // s)
    W = []
    for k in range(N + 1):
        gl = gam[k - 1] if k >= 1 else 0
        gr = gam[k] if k < N else 0
        W.append(U[k * s + 1] + max(gl, gr))
    return finalize(W)


def nfold_plan(n):
    """Binary-doubling plan: list of (i, j, k) meaning law_k = conv(law_i, law_j), ending with k = n.

    Untrusted like the rest of this module: in CHECK mode bridge.run verifies the labels with the trusted
    exact.check_nfold_plan (k = i + j, no label reused, last k = n) before using them.
    """
    plan = []
    have = [1]
    k = 1
    while 2 * k <= n:
        plan.append((k, k, 2 * k))
        have.append(2 * k)
        k *= 2
    acc = None
    for kk in sorted(have, reverse=True):
        if acc is None:
            acc = kk
        elif acc + kk <= n:
            plan.append((acc, kk, acc + kk))
            acc += kk
    need(acc == n, "nfold plan")
    return plan
