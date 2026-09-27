"""Exact checks that need only the Python standard library.

Every function here works with Python integers and ``fractions.Fraction``; nothing is rounded.  This module
is shared by the full checker (``check.py``) and by the standard-library-only family verifier
(``check_family_stdlib.py``), so each inequality has exactly one implementation.

Staircase laws
--------------
A staircase law on a grid of spacing h with N bins is a pair ``(atom, m)`` of nonnegative integers in units of
2^-P (P = 48) with ``m[0] >= m[1] >= ... >= m[N-1]`` and ``atom + sum(m) == 2^P``.  It is the law of |X| for a
symmetric unimodal X: |X| has an atom ``atom`` at 0 and density ``m[k]/h`` on the bin [kh, (k+1)h].  Its CDF is
piecewise linear with knot values ``C[k] = atom + m[0] + ... + m[k-1]`` and equals 1 from Nh on.

Checks raise ``CertificateError`` on failure and otherwise return exact slack values (for reporting only).

Revision notes: ``check_nfold_plan`` verifies the convolution labels of the n-fold sum; ``check_family`` tests
2 A eps0 <= 1 explicitly (it is also implied by (C1) and (C2) at t = 0); ``validate`` accepts only genuine ``int``
values.
"""
import hashlib
import math
from fractions import Fraction

P = 48
ONE = 1 << P


class CertificateError(Exception):
    """A certificate inequality failed (or an input is malformed)."""


def need(cond, msg):
    if not cond:
        raise CertificateError(msg)


# ----------------------------------------------------------------------------------------------- laws
def validate(atom, m, name):
    """Staircase validity: nonnegative, nonincreasing masses, total mass exactly 2^P."""
    need(type(atom) is int and atom >= 0, f"{name}: bad atom")
    need(all(type(v) is int for v in m), f"{name}: non-integer mass")
    need(len(m) > 0 and m[-1] >= 0, f"{name}: negative mass")
    need(all(m[k] >= m[k + 1] for k in range(len(m) - 1)), f"{name}: masses increase")
    need(atom + sum(m) == ONE, f"{name}: total mass != 2^{P}")


def cdf_knots(atom, m):
    C = [atom]
    for v in m:
        C.append(C[-1] + v)
    return C


def cdf_index(C, m, q):
    """CDF (in units 2^-P, as a Fraction) at index position q = t/h >= 0 (piecewise-linear interpolation)."""
    k = q.numerator // q.denominator
    if k >= len(m):
        return Fraction(ONE)
    return C[k] + m[k] * (q - k)


def law_bytes(atom, m):
    return b"".join(int(v).to_bytes(7, "little") for v in [atom] + list(m))


def law_sha256(atom, m):
    return hashlib.sha256(law_bytes(atom, m)).hexdigest()


# ----------------------------------------------------------------------------------------------- polynomials
def polymul(a, b):
    """Exact product of polynomials with nonnegative integer coefficients (Kronecker substitution)."""
    if not a or not b:
        return []
    need(min(a) >= 0 and min(b) >= 0, "polymul: negative coefficient")
    bound = max(a) * max(b) * min(len(a), len(b))  # every product coefficient is <= bound
    nbytes = (bound.bit_length() + 8) // 8
    A = int.from_bytes(b"".join(x.to_bytes(nbytes, "little") for x in a), "little")
    B = int.from_bytes(b"".join(x.to_bytes(nbytes, "little") for x in b), "little")
    L = len(a) + len(b) - 1
    raw = (A * B).to_bytes(L * nbytes, "little")
    return [int.from_bytes(raw[i * nbytes:(i + 1) * nbytes], "little") for i in range(L)]


# ----------------------------------------------------------------------------------------------- min2
def check_min2(X, Z):
    """CDF_Z >= exact CDF of min(|X1|, |X2|), X1, X2 iid ~ X, at every t (same grid).

    On bin k with sigma in [0,1] the exact CDF is 1 - (1 - C_k - m_k sigma)^2, so in units 2^-2P the slack is
    Delta(sigma) = d0 + d1 sigma + m_k^2 sigma^2 (convex); checked at both ends and at the vertex.
    """
    (a, m), (z0, zm) = X, Z
    N = len(m)
    need(len(zm) == N, "min2: grid mismatch")
    C = cdf_knots(a, m)
    Zc = cdf_knots(z0, zm)
    worst = None
    for k in range(N):
        mk = m[k]
        d0 = ONE * Zc[k] - (ONE * ONE - (ONE - C[k]) ** 2)
        d1 = ONE * zm[k] - 2 * (ONE - C[k]) * mk
        d2 = mk * mk
        need(d0 >= 0, f"min2: knot {k}")
        need(d0 + d1 + d2 >= 0, f"min2: knot {k + 1}")
        if d1 < 0 < d2 and -d1 < 2 * d2:  # vertex sigma* = -d1/(2 d2) in (0,1)
            need(d1 * d1 <= 4 * d0 * d2, f"min2: interior of bin {k}")
        worst = d0 if worst is None else min(worst, d0)
    return Fraction(worst, ONE * ONE)


# ----------------------------------------------------------------------------------------------- conv
def conv_E(p, q):
    """E[k], k = 0..N: 2h x (density of |R1 + e R2| at kh), R1, R2 with bin masses p, q and e a fair sign.

    The densities of R1 + R2 and R1 - R2 are piecewise linear with knot values
    Dsum(k) = sum_{j+l=k-1} p_j q_l / h and Ddiff(k) = sum_{j-l=k} p_j q_l / h, and
    E[k] = h (Dsum(k) + Ddiff(k) + Ddiff(-k)).
    """
    N = len(p)
    S = polymul(p, q)  # S[k-1] = sum_{j+l=k-1} p_j q_l
    T = polymul(p, q[::-1])  # T[k+N-1] = sum_{j-l=k} p_j q_l

    def dsum(k):
        return S[k - 1] if 1 <= k <= 2 * N - 1 else 0

    def ddiff(k):
        i = k + N - 1
        return T[i] if 0 <= i <= 2 * N - 2 else 0

    return [dsum(k) + ddiff(k) + ddiff(-k) for k in range(N + 1)]


def conv_exact_pieces(X, Y):
    """Exact CDF of |X+Y| (X, Y independent symmetric staircase laws, same grid) on [0, Nh].

    Returns (Q4, lin, quad): on bin k, 4 * 2^{2P} * CDF(kh + sigma h) = Q4[k] + lin[k] sigma + quad[k] sigma^2.
    """
    (a, p), (b, q) = X, Y
    N = len(p)
    need(len(q) == N, "conv: grid mismatch")
    E = conv_E(p, q)
    Q4 = [4 * a * b]
    lin, quad = [], []
    for k in range(N):
        lk = 4 * a * q[k] + 4 * b * p[k] + 2 * E[k]
        qk = E[k + 1] - E[k]
        lin.append(lk)
        quad.append(qk)
        Q4.append(Q4[-1] + lk + qk)
    return Q4, lin, quad


def check_conv(X, Y, Z):
    """CDF_Z >= exact CDF of |X+Y| at every t in [0, Nh] (and CDF_Z = 1 beyond)."""
    Q4, lin, quad = conv_exact_pieces(X, Y)
    z0, zm = Z
    N = len(lin)
    need(len(zm) == N, "conv: output grid mismatch")
    Zc = cdf_knots(z0, zm)
    worst = None
    for k in range(N):
        d0 = 4 * ONE * Zc[k] - Q4[k]
        d1 = 4 * ONE * zm[k] - lin[k]
        d2 = -quad[k]  # slack Delta(sigma) = d0 + d1 sigma + d2 sigma^2
        need(d0 >= 0, f"conv: knot {k}")
        need(d0 + d1 + d2 >= 0, f"conv: knot {k + 1}")
        if d1 < 0 < d2 and -d1 < 2 * d2:
            need(d1 * d1 <= 4 * d0 * d2, f"conv: interior of bin {k}")
        worst = d0 if worst is None else min(worst, d0)
    return Fraction(worst, 4 * ONE * ONE)


def check_nfold_plan(plan, n):
    """Verify the labels of an n-fold summation plan (trusted; the plan itself comes from untrusted code).

    ``plan`` is a list of steps (i1, i2, k), each meaning "the law labelled k bounds the sum of the independent sums
    labelled i1 and i2", starting from the single label 1 (the branch law Y).  The check enforces that both inputs
    are already available, that k = i1 + i2, that no label is produced twice, and that the last step produces the
    label n.  By induction every label k then stands for a sum of exactly k independent copies of Y, so the law
    labelled n (the output of the last step) bounds the n-branch sum; the caller uses exactly that law as nu_{j+1}.
    """
    need(type(n) is int and n >= 2, "plan: n must be an integer >= 2")
    need(isinstance(plan, (list, tuple)) and len(plan) > 0, "plan: empty")
    have = {1}
    for step in plan:
        need(isinstance(step, tuple) and len(step) == 3 and all(type(v) is int for v in step), "plan: malformed step")
        i1, i2, k = step
        need(i1 in have and i2 in have, f"plan: step {step} uses an unavailable partial sum")
        need(k == i1 + i2, f"plan: step {step} has k != i1 + i2")
        need(k not in have, f"plan: step {step} overwrites label {k}")
        have.add(k)
    need(plan[-1][2] == n, f"plan: last step produces {plan[-1][2]} summands, need n = {n}")
    return True


# ----------------------------------------------------------------------------------------------- coarsen
def check_coarsen(old, new):
    """CDF of `new` on grid 2h >= CDF of `old` on grid h everywhere (both piecewise linear; old knots suffice)."""
    (a0, m0), (a1, m1) = old, new
    N = len(m0)
    need(len(m1) == N, "coarsen: bin count changed")
    C0 = cdf_knots(a0, m0)
    C1 = cdf_knots(a1, m1)
    for k in range(N + 1):
        j, r = divmod(k, 2)
        lhs = 2 * C1[j] if r == 0 else C1[j] + C1[j + 1]
        need(lhs >= 2 * C0[k], f"coarsen: old knot {k}")
    return True


# ----------------------------------------------------------------------------------------------- branch output
def check_branch_output(Y, U, s):
    """CDF_Y(u_i) >= U_{i+1}/2^P for every fine point u_i = i h/s, i = 0..Ns-1.

    With U_{i+1} >= P(f(A,B) <= u_{i+1}) this gives CDF_Y(t) >= P(f <= t) for all t: for t in [u_i, u_{i+1}],
    P(f <= t) <= P(f <= u_{i+1}) <= CDF_Y(u_i) <= CDF_Y(t); and CDF_Y = 1 beyond Nh.
    """
    ya, ym = Y
    N = len(ym)
    need(len(U) >= N * s + 1, "branch: U too short")
    C = cdf_knots(ya, ym)
    worst = None
    for i in range(N * s):
        k, r = divmod(i, s)
        d = s * C[k] + r * ym[k] - s * U[i + 1]
        need(d >= 0, f"branch output: fine point {i}")
        worst = d if worst is None else min(worst, d)
    return Fraction(worst, s * ONE)


# ----------------------------------------------------------------------------------------------- family (Theorem 1A)
def check_family(n, lam, A, eps0, hc, Nx, Xc, G):
    """Conditions (C1), (C2) of Theorem 1A, exactly, on [0, lam * x_max] (index units tau = t / hc).

    Xc = (0, m_c): atomless staircase law of X_c on grid hc with support in the first Nx bins.
    G[k] (k = 1..n): staircase laws on the same grid, G[k] >= CDF of |sum_{i<=k} e_i min(X_i1, X_i2)| (checked
    separately with check_min2 / check_conv).  With F_c the CDF of X_c, g_k the right derivative of G_k,
    H_k = G_k + n eps0 g_k, Ht_k = max_{k' >= k} H_k', Ht_0 = 1:
      (C1) s(t) := F_c(t/lam) - G_n(t) >= 0;
      (C2) s(t) + eps0 (A/lam)(1 - F_c(t/lam))
             >= eps0 [ n g_n(t) + sum_{k=1..n} C(n,k) (2A)^k eps0^{k-1} (Ht_{n-k}(t) - G_n(t)) ].
    On every elementary interval between consecutive knots of F_c(./lam) and all G_k the left side is linear and
    the right side convex (slopes constant on the open interval), so both endpoints suffice; at the right
    endpoint the interval's own slope is used, which is >= the true right derivative (G_k concave), hence
    conservative.
    """
    lam, A, eps0, hc = Fraction(lam), Fraction(A), Fraction(eps0), Fraction(hc)
    need(lam > 1 and A > 0 and eps0 > 0 and hc > 0, "family: parameters out of range")
    need(2 * A * eps0 <= 1, "family: 2 A eps0 > 1")  # Def. 4.4; also implied by (C1), (C2) at t = 0
    xa, xm = Xc
    need(xa == 0, "family: X_c must be atomless")
    validate(xa, xm, "X_c")
    need(all(v == 0 for v in xm[Nx:]), "family: X_c has mass beyond Nx")
    need(sorted(G) == list(range(1, n + 1)), "family: need G_1..G_n")
    for k in G:
        validate(G[k][0], G[k][1], f"G_{k}")
    Cc = cdf_knots(0, xm)
    CG = {k: cdf_knots(*G[k]) for k in G}
    mG = {k: G[k][1] for k in G}
    NG = len(G[1][1])
    tmax = lam * Nx
    knots = {Fraction(k) for k in range(int(tmax) + 1)} | {lam * k for k in range(Nx + 1)}
    knots = sorted(t for t in knots if t <= tmax)
    epsI = eps0 / hc
    binom = [math.comb(n, k) for k in range(n + 1)]
    min_s = min_c2 = None
    where_c2 = None
    for ta, tb in zip(knots[:-1], knots[1:]):
        kb = ta.numerator // ta.denominator
        sl = {k: (mG[k][kb] if kb < NG else 0) for k in G}  # slope per index unit, units 2^-P
        for t in (ta, tb):
            Fc = cdf_index(Cc, xm, t / lam) / ONE
            Gt = {k: cdf_index(CG[k], mG[k], t) / ONE for k in G}
            s = Fc - Gt[n]
            H = {k: Gt[k] + n * epsI * Fraction(sl[k], ONE) for k in G}
            rhs = n * Fraction(sl[n], ONE) / hc
            for k in range(1, n + 1):
                lo = n - k
                Ht = Fraction(1) if lo == 0 else max(H[j] for j in range(lo, n + 1))
                rhs += binom[k] * (2 * A) ** k * eps0 ** (k - 1) * (Ht - Gt[n])
            margin = s + eps0 * (A / lam) * (1 - Fc) - eps0 * rhs
            if min_s is None or s < min_s:
                min_s = s
            if min_c2 is None or margin < min_c2:
                min_c2, where_c2 = margin, t
            need(s >= 0, f"(C1) fails at t = {float(t * hc)}")
            need(margin >= 0, f"(C2) fails at t = {float(t * hc)}")
    return {"C1_min_s": min_s, "C2_min_margin": min_c2, "C2_min_margin_over_eps0": min_c2 / eps0,
            "C2_argmin_t": where_c2 * hc, "n_intervals": len(knots) - 1}


# ----------------------------------------------------------------------------------------------- entry (Theorem 1B)
def check_entry(nu, h, Xc, hc, Nx, L0, alpha0):
    """CDF_nu(t) <= alpha0 + (1 - alpha0) F_c(t / L0) for all t (exact over the union of knots).

    Both sides are piecewise linear with knots at multiples of h and at L0 * hc * k; beyond t = L0 * Nx * hc the
    right side is 1.  Returns the maximum of (left - right) over the knots (<= 0 on success).
    """
    a, m = nu
    C = cdf_knots(a, m)
    xa, xm = Xc
    Cc = cdf_knots(0, xm)
    h, hc, L0, alpha0 = Fraction(h), Fraction(hc), Fraction(L0), Fraction(alpha0)
    tmax = L0 * Nx * hc
    pts = {k * h for k in range(int(tmax / h) + 1)} | {L0 * hc * k for k in range(Nx + 1)}
    worst = None
    for t in sorted(pts):
        if t > tmax:
            continue
        lhs = cdf_index(C, m, t / h) / ONE
        rhs = alpha0 + (1 - alpha0) * cdf_index(Cc, xm, t / (L0 * hc)) / ONE
        d = lhs - rhs
        worst = d if worst is None else max(worst, d)
        need(d <= 0, f"entry fails at t = {float(t)}")
    return worst


# ----------------------------------------------------------------------------------------------- formatting
def floor_sci(x, digits=6):
    """Decimal string d.ddddde<k> whose value is <= the positive Fraction x (a certified floor)."""
    x = Fraction(x)
    need(x > 0, "floor_sci: nonpositive")
    k = len(str(x.numerator)) - len(str(x.denominator))
    while Fraction(10) ** k > x:
        k -= 1
    while Fraction(10) ** (k + 1) <= x:
        k += 1
    mant = x / Fraction(10) ** (k - digits + 1)
    q = mant.numerator // mant.denominator
    s = str(q)
    return f"{s[0]}.{s[1:]}e{k:+d}"


def ceil_sci(x, digits=6):
    """Decimal string whose value is >= the positive Fraction x (a certified ceiling)."""
    x = Fraction(x)
    need(x > 0, "ceil_sci: nonpositive")
    k = len(str(x.numerator)) - len(str(x.denominator))
    while Fraction(10) ** k > x:
        k -= 1
    while Fraction(10) ** (k + 1) <= x:
        k += 1
    mant = x / Fraction(10) ** (k - digits + 1)
    q = -((-mant.numerator) // mant.denominator)
    s = str(q)
    if len(s) > digits:  # rounding up reached the next power of ten
        return f"1.{'0' * (digits - 1)}e{k + 1:+d}"
    return f"{s[0]}.{s[1:]}e{k:+d}"
