#!/usr/bin/env python3
"""Validation of the Part II machinery with the standard library only (NOT part of the proof).

  python3 partII/selftest_stdlib.py [--family partII/inputs/family_n8.json --n 8]

1. Interval module: brackets and directed scaling; the rational constants C0_LO, C0_UP, LN2H_Q_UP; erf enclosures
   against math.erf (float reference, 1e-15).
2. polymul (Kronecker substitution) equals the naive product.
3. The exact CDF formula of |X+Y| (exact.conv_exact_pieces) equals a brute-force sum of rectangle areas.
4. Small grid (e = 5, N = 400, beta = 5/4): every constructor output passes its exact check (gauss, coarsen, min2,
   conv, branch with the pure-Python kernel); the branch fine bounds lie above a float quadrature reference.
5. Tamper tests: strictly more spread laws are rejected by check_conv, check_min2, check_branch_output and
   check_coarsen; nu_0 built for beta = 5/4 passes at beta = 13/10 and fails at beta = 6/5.
6. Plan labels: exact.check_nfold_plan accepts construct.nfold_plan(n) for n = 2..64 and rejects plans
   with k != i1 + i2, an unavailable input, a reused label, or a last label != n.
7. Input typing: floats, booleans and NaN are rejected where integers or rationals are required.
8. Optional (--family): the family verifies, and lam x 1.1 or eps0 x 10 is rejected.
"""
import argparse
import json
import math
import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mkcert import branch, common, constants, construct, exact, gauss  # noqa: E402
from mkcert import ivdec as V  # noqa: E402
from mkcert.exact import ONE, CertificateError  # noqa: E402


def expect_fail(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except CertificateError:
        return True
    raise AssertionError(f"tamper test not rejected: {getattr(fn, '__name__', fn)}")


def rand_law(rng, N, with_atom):
    w = sorted((rng.randrange(1, 1000) for _ in range(N)), reverse=True)
    atom_w = rng.randrange(0, 300) if with_atom else 0
    tot = sum(w) + atom_w
    m = [x * ONE // tot for x in w]
    return ONE - sum(m), m


def brute_abs_sum_cdf(X, Y, t):
    """P(|X+Y| <= t) for independent symmetric staircase X, Y on the unit grid (exact rectangle areas)."""
    def comps(law):
        a, m = law
        out = [(Fraction(a, ONE), None)]
        for j, mj in enumerate(m):
            for sgn in (1, -1):
                lo_, hi_ = (j, j + 1) if sgn == 1 else (-j - 1, -j)
                out.append((Fraction(mj, 2 * ONE), (Fraction(lo_), Fraction(hi_))))
        return out

    def G(w):
        return w * w / 2 if w > 0 else Fraction(0)

    def F_sum(I1, I2, z):
        (a1, b1), (a2, b2) = I1, I2
        return (G(z - a1 - a2) - G(z - b1 - a2) - G(z - a1 - b2) + G(z - b1 - b2)) / ((b1 - a1) * (b2 - a2))

    def F_one(I, z):
        a1, b1 = I
        return min(max((z - a1) / (b1 - a1), Fraction(0)), Fraction(1))

    tot = Fraction(0)
    for w1, I1 in comps(X):
        for w2, I2 in comps(Y):
            if w1 == 0 or w2 == 0:
                continue
            if I1 is None and I2 is None:
                pr = Fraction(1)
            elif I1 is None or I2 is None:
                I = I1 if I2 is None else I2
                pr = F_one(I, t) - F_one(I, -t)
            else:
                pr = F_sum(I1, I2, t) - F_sum(I1, I2, -t)
            tot += w1 * w2 * pr
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    # 1. intervals
    x = V.IV.of(Fraction(1, 3))
    assert x.lo_frac() < Fraction(1, 3) < x.hi_frac()
    assert V.ceil_scaled(x, 10) == 342 and V.floor_scaled(x, 10) == 341
    assert V.ceil_scaled(V.IV.of(5), 0) == 5 and V.floor_scaled(V.IV.of(5), 0) == 5
    constants.verify_rational_constants()
    for q in (Fraction(1, 7), Fraction(3, 2), Fraction(11, 2), Fraction(69, 10), Fraction(71, 10), Fraction(9)):
        e = V.erf(q)
        ref = math.erf(q.numerator / q.denominator)
        assert e.lo_frac() - Fraction(1, 10 ** 15) <= Fraction(ref) <= e.hi_frac() + Fraction(1, 10 ** 15), q
        width = e.hi_frac() - e.lo_frac()
        assert width < (Fraction(1, 10 ** 50) if q <= 7 else Fraction(1, 10 ** 20)), q  # x > 7: tail bound
    print("1. interval brackets, rational constants and erf enclosures ok", flush=True)
    # 2. polymul
    rng = random.Random(1)
    for _ in range(30):
        p = [rng.randrange(1 << 48) for _ in range(rng.randrange(1, 60))]
        q = [rng.randrange(1 << 48) for _ in range(rng.randrange(1, 60))]
        ref = [0] * (len(p) + len(q) - 1)
        for i, u in enumerate(p):
            for j, v in enumerate(q):
                ref[i + j] += u * v
        assert exact.polymul(p, q) == ref
    print("2. polymul ok", flush=True)
    # 3. exact |X+Y| CDF against brute force
    for trial in range(3):
        X = rand_law(rng, 7, trial != 1)
        Y = rand_law(rng, 7, trial != 2)
        Q4, lin, quad = exact.conv_exact_pieces(X, Y)
        for k in range(7):
            for sig in (Fraction(0), Fraction(1, 3), Fraction(4, 5)):
                formula = (Q4[k] + lin[k] * sig + quad[k] * sig * sig) / (4 * ONE * ONE)
                assert formula == brute_abs_sum_cdf(X, Y, k + sig), "conv formula mismatch"
    print("3. exact |X+Y| CDF formula equals brute-force rectangle areas", flush=True)
    # 4. small grid
    e, N, beta = 5, 400, Fraction(5, 4)
    h = Fraction(1, 2 ** e)
    nu = construct.gauss_law(beta, e, N)
    exact.validate(*nu.pair(), "nu0")
    gauss.check_gauss(*nu.pair(), beta, e)
    c = construct.coarsen(nu)
    exact.check_coarsen(nu.pair(), c.pair())
    M2 = construct.min2(nu)
    exact.check_min2(nu.pair(), M2.pair())
    Z = construct.conv(nu, nu)
    exact.check_conv(nu.pair(), nu.pair(), Z.pair())
    U, info = branch.fine_bounds(nu.atom, nu.m, e, 4, 4, 14, 12, kernel="python")
    Y = construct.branch_from_bounds(U, N, 4)
    exact.check_branch_output(Y.pair(), U, 4)
    Cf = [v / ONE for v in nu.cdf()]
    mf = [v / ONE for v in nu.m]
    hf = float(h)

    def FA(xx):
        k = int(xx / hf)
        return 1.0 if k >= N else Cf[k] + mf[k] * (xx / hf - k)

    sub = 32
    worst = 1.0
    for i in range(0, N * 4, 97):
        u = i * hf / 4
        ref = nu.atom / ONE
        for k in range(N):
            for j in range(sub):
                b = (k + (j + 0.5) / sub) * hf
                wgt = mf[k] / sub
                if b <= u:
                    ref += wgt
                else:
                    g = math.atanh(min(math.tanh(u) / math.tanh(b), 1.0)) if b > 0 else 1e9
                    ref += wgt * FA(g)
        worst = min(worst, U[i] / ONE - ref)
    assert worst > -1e-6, worst
    print(f"4. constructor outputs pass all exact checks; min(U - float reference) = {worst:.2e}", flush=True)
    # 5. tamper tests
    Z4 = construct.conv(Z, Z)
    expect_fail(exact.check_conv, nu.pair(), nu.pair(), Z4.pair())
    expect_fail(exact.check_min2, nu.pair(), construct.conv(M2, M2).pair())
    expect_fail(exact.check_branch_output, construct.conv(Y, Y).pair(), U, 4)
    expect_fail(exact.check_coarsen, nu.pair(), construct.conv(c, c).pair())
    gauss.check_gauss(*nu.pair(), Fraction(13, 10), e)
    expect_fail(gauss.check_gauss, *nu.pair(), Fraction(6, 5), e)
    print("5. tamper tests rejected as expected", flush=True)
    # 6. plan labels
    for n in range(2, 65):
        exact.check_nfold_plan(construct.nfold_plan(n), n)
    for plan, n in (([(1, 1, 2), (2, 1, 4)], 4), ([(1, 1, 2), (2, 2, 4)], 3), ([(1, 1, 2), (4, 1, 5)], 5),
                    ([(1, 1, 2), (1, 1, 2), (2, 2, 4)], 4), ([], 2), ([(1, 1, 2), (2, 2, 4)], 8),
                    ([(1, 1, 2.0)], 2), ([(1, 1, 2), (2, 1, 3)], 4)):
        expect_fail(exact.check_nfold_plan, plan, n)
    print("6. n-fold plan labels verified for n = 2..64; 8 tampered plans rejected", flush=True)
    # 7. input typing
    for bad in (1.0, True, "3"):
        expect_fail(common.as_int, bad, "x")
    for bad in (0.5, True, "0.5", "1e3", "nan", "1/0"):
        expect_fail(common.frac, bad, "x")
    expect_fail(common.law_from_json, {"atom": 0, "m": [ONE * 1.0]}, "law")
    expect_fail(common.law_from_json, {"atom": 0, "m": [ONE], "sha256": "0" * 64}, "law")
    assert common.law_from_json({"atom": 0, "m": [ONE], "sha256": exact.law_sha256(0, [ONE])}, "law") == (0, [ONE])
    expect_fail(json.loads, '{"x": NaN}', parse_constant=common._reject_constant)  # the hook used by load_json
    print("7. floats, booleans, NaN and wrong per-law hashes are rejected", flush=True)
    # 8. family
    if a.family:
        from mkcert.family import load_family, verify_family
        fam = load_family(a.family, a.n)
        verify_family(fam)
        for key, factor in (("lam", Fraction(11, 10)), ("eps0", Fraction(10))):
            bad = dict(fam)
            bad[key] = fam[key] * factor
            expect_fail(verify_family, bad)
        print("8. family verifies; lam x 1.1 and eps0 x 10 rejected", flush=True)
    print("PART II STDLIB SELFTEST PASSED", flush=True)


if __name__ == "__main__":
    main()
