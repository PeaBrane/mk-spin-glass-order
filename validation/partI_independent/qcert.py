"""Independent exact verifier for Part I certificates (Prop A, Lemma T, Transfer-Theorem constants).

Written from the mathematics only (the statements of Proposition A, Lemma T and the transfer theorem in the paper); standard library only.
All decisions are exact comparisons of fractions.Fraction values. Transcendentals (pi, exp, ln 2) are enclosed by
series with explicit remainder bounds; every carried quantity is rounded outward to a P-bit mantissa.
"""

import json
import math
import sys
from fractions import Fraction as Q

P = 256  # mantissa bits kept after each outward rounding


def _round(q, upward):
    q = Q(q)
    if q == 0:
        return Q(0)
    if q < 0:
        return -_round(-q, not upward)
    a, b = q.numerator, q.denominator
    s = P - (a.bit_length() - b.bit_length())
    if s >= 0:
        m, rem = divmod(a << s, b)
        if upward and rem:
            m += 1
        return Q(m, 1 << s)
    m, rem = divmod(a, b << (-s))
    if upward and rem:
        m += 1
    return Q(m << (-s))


def up(q):
    return _round(q, True)


def dn(q):
    return _round(q, False)


TINY = Q(1, 1 << (P + 64))


def exp_neg(x):
    """(lo, hi) with lo <= exp(-x) <= hi, for rational x >= 0."""
    x = Q(x)
    if x < 0:
        raise ValueError("exp_neg needs x >= 0")
    if x == 0:
        return Q(1), Q(1)
    s = 0
    while x > Q(1 << s, 16):
        s += 1
    y = x / (1 << s)  # 0 < y <= 1/16: alternating series with decreasing terms
    term = Q(1)
    S = Q(1)
    j = 0
    while True:
        j += 1
        term = -term * y / j
        S += term
        if j % 2 == 1 and -term < TINY:
            lo, hi = S, S - term  # S_j (j odd) <= e^{-y} <= S_{j-1}
            break
    lo, hi = dn(lo), up(hi)
    for _ in range(s):
        lo, hi = dn(lo * lo), up(hi * hi)
    return lo, hi


def ln2_bounds(m=P + 64):
    S = Q(0)
    for k in range(1, m + 1):
        S += Q(1, k << k)
    return dn(S), up(S + Q(1, (m + 1) << m))


def atan_bounds(x):
    """Enclosure of atan(x), 0 < x < 1 rational, by consecutive partial sums of the alternating series."""
    x = Q(x)
    x2 = x * x
    t = x
    S = x
    k = 0
    while True:
        k += 1
        t = t * x2
        prev = S
        S = S + (-1) ** k * t / (2 * k + 1)
        if t / (2 * k + 1) < TINY:
            return min(S, prev), max(S, prev)


def pi_bounds():
    a_lo, a_hi = atan_bounds(Q(1, 5))
    b_lo, b_hi = atan_bounds(Q(1, 239))
    return dn(16 * a_lo - 4 * b_hi), up(16 * a_hi - 4 * b_lo)


def phi_upper():
    """Rational q >= 1/sqrt(2 pi), verified by q^2 * 2 pi_lo >= 1."""
    pi_lo, _ = pi_bounds()
    X = 1 / (2 * pi_lo)
    scale = 1 << P
    num = X * scale * scale
    q = Q(math.isqrt(num.numerator // num.denominator) + 1, scale)
    if not q * q >= X:
        raise AssertionError("phi upper bound failed")
    return q


def eulerian_row(nmax):
    """A[n][k] Eulerian numbers for n <= nmax by the recurrence A(n,k) = (k+1)A(n-1,k) + (n-k)A(n-1,k-1)."""
    A = [[1]]
    for n in range(1, nmax + 1):
        prev = A[-1]
        row = []
        for k in range(n):
            a = (k + 1) * (prev[k] if k < len(prev) else 0)
            b = (n - k) * (prev[k - 1] if 1 <= k <= len(prev) else 0)
            row.append(a + b)
        A.append(row)
    return A


def kappas(gmax):
    """kappa_g = p_2^{*g}(0), p_2(x) = 2(1-2|x|)_+ ; = 2 * IrwinHall_{2g}(g), two independent routes."""
    A = eulerian_row(2 * gmax - 1)
    out = {}
    for g in range(1, gmax + 1):
        m = 2 * g
        via_euler = Q(2 * A[m - 1][g - 1], math.factorial(m - 1))
        via_ih = Q(2 * sum((-1) ** j * math.comb(m, j) * (g - j) ** (m - 1) for j in range(g + 1)),
                   math.factorial(m - 1))
        if via_euler != via_ih:
            raise AssertionError(f"kappa mismatch at g={g}")
        out[g] = via_euler
    for g, v in ((1, Q(2)), (2, Q(4, 3)), (3, Q(11, 10)), (4, Q(302, 315))):
        if out[g] != v:
            raise AssertionError(f"kappa_{g} != {v}")
    return out


KAPPA = kappas(64)
PHI_UP = phi_upper()
LN2_LO, LN2_HI = ln2_bounds()
_EXP_CACHE = {}


def exp_neg_hi(x):
    x = Q(x)
    v = _EXP_CACHE.get(x)
    if v is None:
        v = exp_neg(x)[1]
        _EXP_CACHE[x] = v
    return v


def coefficients(n):
    """c[j] = C(n, n-j) kappa_{n-j}: coefficient of p^j in sum_{g=1}^n C(n,g) kappa_g p^{n-g}."""
    return [math.comb(n, n - j) * KAPPA[n - j] for j in range(n)]


def prop_a(r, w, L, R, n, coef):
    """Upper bounds (Gamma, p, theta) of Prop A at H(r, w) with parameters L, R. Requires w < 1, L, R > 0."""
    if not (w < 1):
        raise CertFail("w >= 1")
    if not (L > 0 and R > 0):
        raise CertFail("L or R not positive")
    rho = up(r / (1 - w))
    p = up(2 * w + (2 * rho * R) ** 2 + 4 * rho * L)
    eL = exp_neg_hi(2 * L)
    eR = exp_neg_hi(2 * R)
    if not eR < 1:
        raise CertFail("tanh R lower bound not positive")
    theta = up((1 + eL) * (1 + eR) / (1 - eR))
    acc = coef[n - 1]
    for j in range(n - 2, -1, -1):
        acc = up(acc * p + coef[j])
    return up(theta * acc), p, theta


class CertFail(Exception):
    pass


def parse_rational(s, what):
    if not isinstance(s, str):
        raise CertFail(f"{what}: not a string")
    try:
        v = Q(s)
    except (ValueError, ZeroDivisionError):
        raise CertFail(f"{what}: not an exact rational/decimal")
    return v


def floor_dec(q, d=6):
    return math.floor(q * 10 ** d) / 10 ** d


def check(cert, n_expected=None):
    if cert.get("kind") != "mk-partI-certificate":
        raise CertFail("kind")
    n = cert["n"]
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise CertFail("n")
    if n_expected is not None and n != n_expected:
        raise CertFail("n mismatch")
    if cert.get("couplings") != "gaussian":
        raise CertFail("couplings")
    beta = parse_rational(cert["beta"], "beta")
    if beta <= 0:
        raise CertFail("beta <= 0")
    claims = cert["claims"]
    coef = coefficients(n)

    r = up(PHI_UP / beta)  # H_0(phi/beta, 0)
    w = Q(0)
    rs, ws, gams = [r], [w], []
    min_one_minus_w = Q(1)
    for i, (Ls, Rs) in enumerate(cert["levels"]):
        L = parse_rational(Ls, f"L_{i}")
        R = parse_rational(Rs, f"R_{i}")
        min_one_minus_w = min(min_one_minus_w, 1 - w)
        G, p, _ = prop_a(r, w, L, R, n, coef)
        r, w = up(r * G), up(p ** n)
        rs.append(r)
        ws.append(w)
        gams.append(G)
    K = len(cert["levels"])
    t = cert["tail"]
    A = parse_rational(t["A"], "A")
    gam = parse_rational(t["gamma"], "gamma")
    L = parse_rational(t["L"], "tail L")
    R = parse_rational(t["R"], "tail R")
    if not (A > 0 and gam > 0):
        raise CertFail("A or gamma not positive")
    rK, wK = rs[-1], ws[-1]
    W = A * rK * rK
    res = {"n": n, "beta": cert["beta"], "claims": claims, "K": K}
    ok = {}
    ok["T0a"] = wK <= W
    ok["T0b"] = W < 1
    if not ok["T0b"]:
        raise CertFail("T0: A r_K^2 >= 1")
    G, p, theta = prop_a(rK, up(W), L, R, n, coef)
    ok["T1_Gamma_le_gamma"] = G <= gam
    ok["T1_gamma_lt_1"] = gam < 1
    pn = up(p ** n)
    rhs = A * gam * gam * rK * rK
    ok["T2"] = pn <= rhs
    res["margins"] = {
        "min_1_minus_w_levels": float(min_one_minus_w),
        "max_Gamma_levels": float(max(gams)) if gams else None,
        "T0_ratio_wK_over_ArK2": float(wK / W),
        "T0_ArK2": float(W),
        "T1_gamma_minus_Gamma": float(gam - G),
        "T1_rel_gamma_minus_Gamma": float((gam - G) / gam),
        "T1_one_minus_gamma": float(1 - gam),
        "T2_ratio_pn_over_rhs": float(pn / rhs),
        "Gamma_K_upper": float(G),
        "r0_upper": float(rs[0]),
        "rK_upper": float(rK),
    }
    closes = all(ok.values())
    res["hypotheses"] = ok
    res["closes"] = closes

    # Transfer-Theorem constants.  Chain k = 0..K with (r_k, w_k); tail k = K+m, m >= 1:
    # (r_K gam^m, W gam^{2m}) by Lemma T.
    ls = Q(0)
    lb = Q(0)
    S = Q(0)
    Pc = Q(0)
    c_b = 1 + 2 * LN2_HI
    for rk, wk in zip(rs, ws):
        ls = up(ls + wk / 2 + 2 * rk)
        lb = up(lb + Q(3, 4) * wk + c_b * rk + wk * wk + wk * rk + rk * rk / 2)
        S = max(S, up(wk + 2 * rk))
        Pc = max(Pc, up(wk + 2 * LN2_HI * rk))
    Wu = up(W)

    def geo(j):  # sum_{m>=1} gam^{jm}
        gj = gam ** j
        return up(gj / (1 - gj))

    if gam < 1:
        ls = up(ls + Wu / 2 * geo(2) + 2 * rK * geo(1))
        lb = up(lb + Q(3, 4) * Wu * geo(2) + c_b * rK * geo(1) + Wu * Wu * geo(4) + Wu * rK * geo(3)
                + rK * rK / 2 * geo(2))
        S = max(S, up(Wu * gam * gam + 2 * rK * gam))
        Pc = max(Pc, up(Wu * gam * gam + 2 * LN2_HI * rK * gam))
    order_margin = 1 - 2 * ls - S
    ea = 1 - 2 * ls
    om_plus = max(order_margin, Q(0))
    eq2 = om_plus * om_plus
    one_m = max(1 - 2 * lb, Q(0))
    imb = one_m * one_m - Pc
    res["constants_upper"] = {"ell_s": float(ls), "ell_b": float(lb), "S": float(S), "P": float(Pc)}
    res["lower_bounds_floor6"] = {
        "pole_1_minus_S": floor_dec(1 - S),
        "EA_fixed_poles_1_minus_2ell_s": floor_dec(ea),
        "EQ2_(1-2ell_s-S)^2": floor_dec(eq2) if order_margin > 0 else None,
        "ED2_(1-2ell_b)^2-P": floor_dec(imb),
    }
    res["order_margin_lower"] = float(order_margin)
    res["imbalance_margin_lower"] = float(imb)
    verdict = {}
    for c in claims:
        if c in ("pole", "bounds"):
            verdict[c] = closes
        elif c == "order":
            verdict[c] = closes and order_margin > 0
        elif c == "imbalance":
            verdict[c] = closes and imb > 0
        else:
            raise CertFail(f"unknown claim {c}")
    res["verdict"] = verdict
    res["all_claims_hold"] = all(verdict.values())
    return res


def _reject_float(s):
    raise CertFail(f"JSON float/constant {s!r} not allowed")


def main(argv):
    out = []
    for path in argv[1:]:
        try:
            with open(path) as fh:
                cert = json.load(fh, parse_float=_reject_float, parse_constant=_reject_float)
            res = check(cert)
        except CertFail as e:
            res = {"error": str(e), "all_claims_hold": False}
        res["file"] = path
        out.append(res)
        print(json.dumps(res), flush=True)
    return out


if __name__ == "__main__":
    main(sys.argv)
