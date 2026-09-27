"""Rational bounds for c0 = (ln 2)/2 and the rigorous Theorem 2/3 constants kappa_Q, kappa_C (trusted code).

For l >= 1 let phi_l := E psi(||K|-|K'||) + E exp(-4 max(|K|,|K'|)) with K, K' iid ~ K_{l-1} and
psi(s) = 1 - (1 - 1/(1+e^{2s}))^2 (paper, Section 4).  If F bounds the CDF of |K_{l-1}| from above and
V ~ F, then P(||K|-|K'|| <= s) <= F(2s) (unimodality) and P(max <= s) <= F(s)^2, so, psi and e^{-4s} being
decreasing,

    E psi(Delta) <= E psi(V/2) = (3/4) a + sum_k m_k (2/h) int_{kh/2}^{(k+1)h/2} psi(s) ds,
    E e^{-4max}   <= E e^{-4 max(V,V')} = a^2 + sum_k e^{-4kh} (2 m_k/h) (C_k I0 + (m_k/h) I1),

for a staircase F (atom a, masses m_k, knots C_k, all in probability units), with
I0 = (1 - e^{-4h})/4, I1 = (1 - (1+4h) e^{-4h})/16 and int psi = Psi, Psi(s) = (sigma(2s) - log(1+e^{-2s}))/2.
All transcendental numbers are replaced by integer upper bounds at scale 2^Q (upper endpoints of mkcert.ivdec
enclosures), so each phi bound is an exact rational.  Family levels (Theorem 1A):
P(|K_{J+i}| <= u) <= alpha_i + f_c u / L_i with L_i = L0 lam^i, alpha_i = A c0 / L_i, giving
phi_{J+1+i} <= a1/L_i + a2/L_i^2 with a1 = (3/4) A c0 + 2 f_c (ln2/2 + 1/4), a2 = A^2 c0^2 + A c0 f_c/2 + f_c^2/8
(rational upper bounds for c0 and ln 2).  Then T_j = sum_{l >= j} phi_l and

    kappa_Q = sum_{j>=1} (2n-1)(2n)^{-j} (1 - 2 T_j)_+,   kappa_C = sum_{j>=1} (2n-1)(2n)^{-j} (1 - T_j)_+,

truncated at a finite j (dropping nonnegative terms), so the returned values are exact rational lower bounds.
"""
from fractions import Fraction

from . import ivdec as V
from .exact import ONE, need

# Rational bounds for c0 = (ln 2)/2 and for ln2/2 + 1/4 = int_0^infty psi (verified by verify_rational_constants).
C0_LO = Fraction(34657359, 10 ** 8)
C0_UP = Fraction(34657360, 10 ** 8)
LN2H_Q_UP = Fraction(59657360, 10 ** 8)


def verify_rational_constants():
    c0 = V.ln2() * V.IV.of(Fraction(1, 2))
    need(C0_LO < c0.lo_frac() and c0.hi_frac() < C0_UP, "c0 bounds")
    need((c0 + Fraction(1, 4)).hi_frac() < LN2H_Q_UP, "ln2/2 + 1/4 bound")
    return {"C0_LO": str(C0_LO), "C0_UP": str(C0_UP), "LN2H_Q_UP": str(LN2H_Q_UP),
            "ln2_enclosure_width": float(V.ln2().hi_frac() - V.ln2().lo_frac())}


Q = 96
_PSI = {}  # s (Fraction) -> enclosure of Psi(s)
_EXP4 = {}  # s (Fraction) -> integer upper bound of e^{-4 s} 2^Q
_TAB = {}


def _Psi(s):
    if s not in _PSI:
        q = V.exp(V.IV.of(-2 * s))
        _PSI[s] = (1 / (1 + q) - V.log(1 + q)) * V.IV.of(Fraction(1, 2))
    return _PSI[s]


def _exp4_up(s):
    if s not in _EXP4:
        _EXP4[s] = V.ceil_scaled(V.exp(V.IV.of(-4 * s)), Q)
    return _EXP4[s]


def _tables(e, n):
    """Integer upper bounds (scale 2^Q): per-bin averages of psi over [kh/2, (k+1)h/2] and e^{-4kh}, k < n."""
    h = Fraction(2) ** (-e)
    if e not in _TAB:
        eh = V.exp(V.IV.of(-4 * h))
        A0 = V.ceil_scaled((1 - eh) * V.IV.of(2 / (4 * h)), Q)
        A1 = V.ceil_scaled((1 - eh * V.IV.of(1 + 4 * h)) * V.IV.of(2 / (16 * h * h)), Q)
        _TAB[e] = ([], [], A0, A1)
    avgpsi, ex, A0, A1 = _TAB[e]
    for k in range(len(avgpsi), n):
        avgpsi.append(V.ceil_scaled((_Psi(h * (k + 1) / 2) - _Psi(h * k / 2)) * V.IV.of(2 / h), Q))
        ex.append(_exp4_up(h * k))
    return avgpsi, ex, A0, A1


def phi_bridge(atom, m, e):
    """Exact rational upper bound of phi_l from a staircase CDF bound (atom, m) of |K_{l-1}| on h = 2^-e."""
    last = max((k + 1 for k in range(len(m)) if m[k] > 0), default=0)
    avgpsi, ex, A0, A1 = _tables(e, last)
    C = atom
    s_psi = 0
    s_exp = 0
    for k in range(last):
        mk = m[k]
        s_psi += mk * avgpsi[k]
        s_exp += ex[k] * (A0 * mk * C + A1 * mk * mk)
        C += mk
    psi_part = Fraction(3 * atom, 4 * ONE) + Fraction(s_psi, ONE << Q)
    exp_part = Fraction(atom * atom, ONE * ONE) + Fraction(s_exp, (ONE * ONE) << (2 * Q))
    return psi_part + exp_part, psi_part, exp_part


def family_tail_coeffs(A, fc):
    a1 = Fraction(3, 4) * A * C0_UP + 2 * fc * LN2H_Q_UP
    a2 = A * A * C0_UP * C0_UP + A * C0_UP * fc / 2 + fc * fc / 8
    return a1, a2


def family_tail(i0, a1, a2, L0, lam):
    """sum_{i >= i0} (a1 / L_i + a2 / L_i^2), L_i = L0 lam^i (exact rational)."""
    return a1 * lam ** (-i0) / (L0 * (1 - 1 / lam)) + a2 * lam ** (-2 * i0) / (L0 * L0 * (1 - lam ** -2))


def kappas(n, phis, J, a1, a2, L0, lam):
    """phis[l-1] = upper bound of phi_l for l = 1..J+1 (bridge levels 0..J).  Returns exact lower bounds."""
    lam = Fraction(lam)
    L0 = Fraction(L0)
    need(len(phis) == J + 1, "kappas: need phi_1..phi_{J+1}")
    T = {}
    acc = family_tail(1, a1, a2, L0, lam)
    for l in range(J + 1, 0, -1):
        acc += phis[l - 1]
        T[l] = acc
    w = Fraction(2 * n - 1, 2 * n)
    kQ = Fraction(0)
    kC = Fraction(0)
    j = 1
    first_Q = first_C = None
    while True:
        Tj = T[j] if j <= J + 1 else family_tail(j - J - 1, a1, a2, L0, lam)
        if Tj < Fraction(1, 2) and first_Q is None:
            first_Q = j
        if Tj < 1 and first_C is None:
            first_C = j
        kQ += w * max(Fraction(0), 1 - 2 * Tj)
        kC += w * max(Fraction(0), 1 - Tj)
        # stop once the remaining weight (2n)^{-j} is below 10^-12 of the accumulated kappa_Q
        if first_Q is not None and j > J + 1 and w * 10 ** 12 < kQ:
            break
        j += 1
        w /= 2 * n
    return {"S_upper": T[1], "T_upper": T, "kappa_Q": kQ, "kappa_C": kC, "j_first_T_below_half": first_Q,
            "j_first_T_below_one": first_C, "j_max": j}
