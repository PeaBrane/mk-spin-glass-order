"""Part I certificates: Prop A, Lemma T and the Transfer-Theorem constants (trusted code; standard library only).

Setting (b = 2 diamond lattice, n branches, couplings J_e iid N(0,1), xi_0 = beta J).  H_k(r, w) means: there is an
event E_k of the couplings of a scale-k sub-diamond with P(E_k^c) <= w and P(xi_k in dx, E_k) <= r dx.

Prop A.  If H_k(r, w) with w < 1, then for any L, R > 0
    rho = r/(1-w),  p = 2w + (2 rho R)^2 + 4 rho L,  theta = (1 + e^{-2L})/tanh R,
    Gamma = theta * sum_{g=1}^{n} C(n,g) kappa_g p^{n-g},        and H_{k+1}(r Gamma, p^n) holds.
    kappa_g = p_2^{*g}(0) with p_2(x) = 2(1 - 2|x|)_+  (kappa_1..4 = 2, 4/3, 11/10, 302/315).
Lemma T.  If H_K(r_K, w_K) and A, gamma, L, R > 0 satisfy
    (T0) w_K <= A r_K^2 and A r_K^2 < 1,  (T1) Gamma(r_K, A r_K^2) <= gamma < 1,  (T2) p(r_K, A r_K^2)^n <= A gamma^2 r_K^2,
    then H_k(r_K gamma^{k-K}, A r_K^2 gamma^{2(k-K)}) for every k >= K.
Base.  H_0(phi/beta, 0) with phi = 1/sqrt(2 pi); H is monotone in (r, w), so larger r, w are always admissible,
and a certificate at beta certifies every beta' >= beta with the same constants.
Constants (Transfer Theorem with the analytic feeder F_k = w_k + 2 r_k t, Q_k = w_k + 4 r_k t):
    ell_s = sum_{k>=0} (w_k/2 + 2 r_k)                                         (spin-only rate, Lemma S)
    ell_b = sum_{k>=0} [(3/4) w_k + (1 + 2 ln2) r_k + w_k^2 + w_k r_k + r_k^2/2] (blue rate, Lemma B')
    S = sup_k (w_k + 2 r_k),  P = sup_k (w_k + 2 ln2 r_k)
    pole E<s_A s_B>^2 >= 1 - S;  fixed-pole EA E<s_x>^2 >= 1 - 2 ell_s;  E Q^2 >= (1 - 2 ell_s - S)_+^2;
    blue imbalance E D^2 >= (1 - 2 ell_b)_+^2 - P.
The tail sums over k >= K use r_k = r_K gamma^{k-K}, w_k = A r_K^2 gamma^{2(k-K)}; the terms w^2, w r, r^2/2 are
summed with the common factor 1/(1 - gamma^2), an upper bound since 1 - gamma^2 <= 1 - gamma^3 <= 1 - gamma^4.

Arithmetic.  Every quantity is an exact Fraction.  The only transcendental numbers, phi = 1/sqrt(2 pi), e^{-2L},
e^{-2R} and ln 2, are replaced by one-sided rational bounds taken from the endpoints of the directed-rounding
decimal enclosures of partII/mkcert/ivdec.py.  The carried bounds r_k, w_k and the intermediate p and Horner sums
are rounded UP to DIGITS significant decimal digits (round_up), which keeps the numbers short and is valid because
every map involved is nondecreasing in these arguments (polynomials with nonnegative coefficients, and H monotone).
No float enters any computation.
"""
import os
import sys
from fractions import Fraction
from math import comb, factorial

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "partII"))

from mkcert import ivdec as V  # noqa: E402  (directed-rounding decimal intervals, shared with Part II)
from mkcert.exact import CertificateError, need  # noqa: E402

DIGITS = 64
CLAIMS = ("pole", "order", "imbalance", "bounds")


# ----------------------------------------------------------------------------------------------- exact helpers
def kappa(g):
    """kappa_g = p_2^{*g}(0) exactly: p_2 is the law of a sum of two U[-1/4, 1/4], so kappa_g = 2 f_{2g}(g) with
    f_m(x) = sum_{k <= x} (-1)^k C(m, k) (x - k)^{m-1} / (m-1)! the Irwin-Hall density."""
    m = 2 * g
    s = sum((-1) ** k * comb(m, k) * (g - k) ** (m - 1) for k in range(g + 1))
    return Fraction(2 * s, factorial(m - 1))


def kappas(n):
    return [None] + [kappa(g) for g in range(1, n + 1)]


def _log10_guess(x):
    """An integer close to log10(x) for a Fraction x > 0 (a starting point only; callers correct it exactly)."""
    return (x.numerator.bit_length() - x.denominator.bit_length()) * 30103 // 100000


def round_up(x, digits=DIGITS):
    """Smallest a * 10^e >= x with integers 0 < a <= 10^digits (x >= 0 a Fraction); exact integer arithmetic."""
    x = Fraction(x)
    need(x >= 0, "round_up: negative argument")
    if x == 0:
        return x
    e = _log10_guess(x) - digits
    while x >= Fraction(10) ** (e + digits):
        e += 1
    while x < Fraction(10) ** (e + digits - 1):
        e -= 1
    q = x / Fraction(10) ** e
    a = -((-q.numerator) // q.denominator)
    return Fraction(a) * Fraction(10) ** e


def floor_dec(x, d=6):
    """Decimal string of the largest multiple of 10^-d that is <= x (a certified lower bound)."""
    v = Fraction(x) * 10 ** d
    k = v.numerator // v.denominator
    sign = "-" if k < 0 else ""
    q, r = divmod(abs(k), 10 ** d)
    return f"{sign}{q}.{r:0{d}d}"


def sci(x, digits=6, up=True):
    """Scientific-notation string >= x (up=True) or <= x (up=False) for x > 0; '0' for x == 0."""
    x = Fraction(x)
    if x == 0:
        return "0"
    neg = x < 0
    a = abs(x)
    k = _log10_guess(a)
    while Fraction(10) ** k > a:
        k -= 1
    while Fraction(10) ** (k + 1) <= a:
        k += 1
    mant = a / Fraction(10) ** (k - digits + 1)
    toward_up = up != neg  # rounding the magnitude up gives an upper bound only for positive x
    q = -((-mant.numerator) // mant.denominator) if toward_up else mant.numerator // mant.denominator
    s = str(q)
    if len(s) > digits:
        s, k = s[:digits], k + 1
    return f"{'-' if neg else ''}{s[0]}.{s[1:]}e{k:+d}"


# ----------------------------------------------------------------------------------------------- transcendentals
def _constants():
    sqrt2pi = V.sqrt(V.IV.of(2) * V.pi())
    phi_hi = 1 / sqrt2pi.lo_frac()  # >= 1/sqrt(2 pi)
    ln2 = V.ln2()
    need(ln2.lo_frac() > Fraction(693147, 10 ** 6) and ln2.hi_frac() < Fraction(693148, 10 ** 6), "ln 2 enclosure")
    need(Fraction(398942, 10 ** 6) < phi_hi < Fraction(398943, 10 ** 6), "phi enclosure")
    return phi_hi, ln2.hi_frac()


PHI_HI, LN2_HI = _constants()
_THETA = {}


def theta_hi(L, R):
    """Rational upper bound of theta = (1 + e^{-2L}) / tanh R, L, R > 0 (tanh R >= (1-u)/(1+u) for u >= e^{-2R})."""
    key = (L, R)
    if key not in _THETA:
        e2L = V.exp(V.IV.of(-2 * L)).hi_frac()
        e2R = V.exp(V.IV.of(-2 * R)).hi_frac()
        need(e2R < 1, "theta: tanh R lower bound not positive")
        _THETA[key] = (1 + e2L) * (1 + e2R) / (1 - e2R)
    return _THETA[key]


# ----------------------------------------------------------------------------------------------- Prop A, Lemma T
def prop_a(n, kap, r, w, L, R):
    """Upper bounds (Gamma_up, p_up) for Prop A at (r, w, L, R); then H_k(r, w) => H_{k+1}(r Gamma_up, p_up^n)."""
    need(r > 0 and 0 <= w < 1, "Prop A: need r > 0 and 0 <= w < 1")
    need(L > 0 and R > 0, "Prop A: need L, R > 0")
    rho = r / (1 - w)
    p = round_up(2 * w + (2 * rho * R) ** 2 + 4 * rho * L)
    poly = n * kap[1]  # Horner in p for sum_{g=1}^{n} C(n,g) kappa_g p^{n-g}
    for g in range(2, n + 1):
        poly = round_up(poly * p + comb(n, g) * kap[g])
    return round_up(theta_hi(L, R) * poly), p


def rat(v, name):
    """An exact rational from a JSON string: 'p/q', an integer or a finite decimal (e.g. '1.52e-12' is 152/10^14).
    JSON numbers with a fraction or exponent (Python floats), booleans, NaN and infinities are rejected."""
    need(type(v) in (str, int), f"{name}: must be a string holding an exact rational (got {type(v).__name__})")
    try:
        return Fraction(v)
    except (ValueError, ZeroDivisionError):
        raise CertificateError(f"{name}: not a finite rational: {v!r}")


def parse(cert):
    """Validate the input schema and return (n, beta, claims, levels, tail) with exact rationals."""
    need(isinstance(cert, dict) and cert.get("kind") == "mk-partI-certificate", "input: wrong kind")
    need(cert.get("couplings") == "gaussian", "input: couplings must be 'gaussian' (density bound 1/sqrt(2 pi))")
    n = cert.get("n")
    need(type(n) is int and n >= 2, "input: n must be a JSON integer >= 2")
    beta = rat(cert.get("beta"), "beta")
    need(beta > 0, "input: beta must be positive")
    claims = cert.get("claims")
    need(isinstance(claims, list) and claims and all(c in CLAIMS for c in claims), f"input: claims must be in {CLAIMS}")
    lv = cert.get("levels")
    need(isinstance(lv, list), "input: levels must be a list")
    levels = []
    for k, pair in enumerate(lv):
        need(isinstance(pair, list) and len(pair) == 2, f"input: level {k} must be [L, R]")
        levels.append((rat(pair[0], f"level {k} L"), rat(pair[1], f"level {k} R")))
    t = cert.get("tail")
    need(isinstance(t, dict) and set(t) == {"A", "gamma", "L", "R"}, "input: tail must have exactly A, gamma, L, R")
    tail = {k: rat(t[k], f"tail {k}") for k in ("A", "gamma", "L", "R")}
    return n, beta, claims, levels, tail


def check(cert):
    """Verify one certificate.  Raises CertificateError on the first failed hypothesis or claim; returns a record
    with exact bounds (as strings) otherwise."""
    n, beta, claims, levels, tail = parse(cert)
    kap = kappas(n)
    r = round_up(PHI_HI / beta)  # H_0(r, 0)
    w = Fraction(0)
    r0 = r
    ell_s = ell_b = S = P = Fraction(0)
    for k, (L, R) in enumerate(levels):
        need(w < 1, f"level {k}: w >= 1")
        S = max(S, w + 2 * r)
        P = max(P, w + 2 * LN2_HI * r)
        ell_s += w / 2 + 2 * r
        ell_b += Fraction(3, 4) * w + (1 + 2 * LN2_HI) * r + w * w + w * r + r * r / 2
        gam, p = prop_a(n, kap, r, w, L, R)
        r, w = round_up(r * gam), round_up(p ** n)
    K = len(levels)
    A, gamma, L, R = tail["A"], tail["gamma"], tail["L"], tail["R"]
    need(A > 0, "tail: A must be positive")
    need(0 < gamma < 1, "tail: need 0 < gamma < 1")
    wT = A * r * r
    need(w <= wT, f"(T0) fails: w_K > A r_K^2 at K = {K}")
    need(wT < 1, "(T0) fails: A r_K^2 >= 1")
    gamT, pT = prop_a(n, kap, r, wT, L, R)
    need(gamT <= gamma, f"(T1) fails: Gamma(r_K, A r_K^2) > gamma at K = {K}")
    lhs2, rhs2 = pT ** n, A * gamma * gamma * r * r
    need(lhs2 <= rhs2, f"(T2) fails: p^n > A gamma^2 r_K^2 at K = {K}")
    g1, g2 = 1 - gamma, 1 - gamma * gamma
    ell_s += A * r * r / (2 * g2) + 2 * r / g1
    ell_b += (Fraction(3, 4) * A * r * r / g2 + (1 + 2 * LN2_HI) * r / g1
              + (A * A * r ** 4 + A * r ** 3 + r * r / 2) / g2)
    S = max(S, wT + 2 * r)
    P = max(P, wT + 2 * LN2_HI * r)
    ms = 1 - 2 * ell_s - S
    mb = 1 - 2 * ell_b
    eq2 = ms * ms if ms > 0 else Fraction(0)
    ed2 = (mb * mb if mb > 0 else Fraction(0)) - P
    out = {
        "n": n, "beta": str(beta), "K": K, "claims": claims,
        "r0_upper": sci(r0), "rK_upper": sci(r), "wK_upper": sci(w), "tail": {k: str(v) for k, v in tail.items()},
        "T1_Gamma_upper": sci(gamT), "T1_margin_gamma_minus_Gamma_lower": sci(gamma - gamT, up=False),
        "T2_ratio_lower": sci(rhs2 / lhs2, up=False),
        "ell_s_upper": sci(ell_s), "ell_b_upper": sci(ell_b), "S_upper": sci(S), "P_upper": sci(P),
        "pole_lower6": floor_dec(1 - S), "EA_fixed_lower6": floor_dec(1 - 2 * ell_s),
        "EQ2_lower6": floor_dec(eq2), "ED2_lower6": floor_dec(ed2),
        "order_margin_1_minus_2ell_s_minus_S_lower6": floor_dec(ms),
        "order_positive": ms > 0, "imbalance_positive": ed2 > 0,
        "digits": DIGITS, "interval_digits": V.PREC,
    }
    if "order" in claims:
        need(ms > 0, "claim 'order' fails: 1 - 2 ell_s - S <= 0")
    if "imbalance" in claims:
        need(ed2 > 0, "claim 'imbalance' fails: (1 - 2 ell_b)_+^2 - P <= 0")
    return out
