#!/usr/bin/env python3
"""Full validation of the certificate machinery (optional; NOT part of the proof).  Needs numpy, numba, scipy
and optionally python-flint (Arb cross-check of the interval tables).  The standard-library subset is
partII/selftest_stdlib.py.

  python partII/optional/selftest_full.py --threads 2 --family partII/inputs/family_n8.json --n 8

1. Arb endpoint conversion is exact and outward (ball(1/3) brackets 1/3; ceil/floor scaled are correct).
2. polymul (Kronecker) equals the naive product.
3. Small-grid laws: every constructor output passes its exact check, and float references agree with the bounds
   (bound - reference >= 0 up to float noise).
4. Kernel cross-check: the numba and numpy kernels give bit-identical fine bounds U.
5. Tamper tests: deliberately weakened laws / parameters must be rejected by the checks.
"""
import argparse
import math
import os
import random
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # arb_tools.py
sys.path.insert(0, os.path.dirname(HERE))  # partII: the mkcert package

from mkcert import branch, constants, construct, exact, gauss  # noqa: E402
from mkcert import ivdec as V  # noqa: E402
from mkcert.exact import ONE, CertificateError  # noqa: E402


def expect_fail(fn, *args):
    try:
        fn(*args)
    except CertificateError:
        return True
    raise AssertionError(f"tamper test not rejected: {fn.__name__}")


def cdf_float(law, h, t):
    C = np.array(law.cdf(), dtype=float) / ONE
    m = np.array(law.m, dtype=float) / ONE
    k = np.minimum((t / h).astype(np.int64), law.N)
    kk = np.minimum(k, law.N - 1)
    return np.where(k >= law.N, 1.0, C[np.minimum(k, law.N)] + m[kk] * (t / h - k))


def rand_law(rng, N, with_atom):
    """Random staircase law (atom, nonincreasing masses) with total exactly 2^P."""
    w = sorted((rng.randrange(1, 1000) for _ in range(N)), reverse=True)
    atom_w = rng.randrange(0, 300) if with_atom else 0
    tot = sum(w) + atom_w
    m = [x * ONE // tot for x in w]
    atom = ONE - sum(m)
    return atom, m


def brute_abs_sum_cdf(X, Y, t):
    """P(|X+Y| <= t) for independent symmetric staircase X, Y on the unit grid, by summing exact rectangle
    areas over all pairs of signed bins (independent of the polynomial formula in exact.conv_exact_pieces)."""
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

    def F_sum(I1, I2, z):  # P(U1 + U2 <= z), U_i uniform on I_i
        (a1, b1), (a2, b2) = I1, I2
        return (G(z - a1 - a2) - G(z - b1 - a2) - G(z - a1 - b2) + G(z - b1 - b2)) / ((b1 - a1) * (b2 - a2))

    def F_one(I, z):  # P(U <= z)
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
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--family", default=None, help="optional family file for family tamper tests")
    ap.add_argument("--n", type=int, default=None, help="n of --family (required with --family)")
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    # 1. interval module: brackets, directed scaling, and agreement with Arb (python-flint) if installed
    x = V.IV.of(Fraction(1, 3))
    assert x.lo_frac() < Fraction(1, 3) < x.hi_frac(), "interval does not bracket 1/3"
    assert V.ceil_scaled(x, 10) == 342 and V.floor_scaled(x, 10) == 341
    assert V.ceil_scaled(V.IV.of(5), 0) == 5 and V.floor_scaled(V.IV.of(5), 0) == 5
    constants.verify_rational_constants()
    try:
        import arb_tools as AT
    except ImportError:
        AT = None
    if AT is not None:
        AT.verify_constants()
        for (e_, s_, R_, n_) in ((7, 4, 14, 1537), (9, 4, 14, 3000)):
            Dup, Dlo = branch.delta_tables(e_, s_, R_, n_)
            stp = AT.arb(1) / (AT.arb(2) ** e_ * s_)
            for k in range(1, n_ + 1):
                dv = -((1 - (-(2 * stp * k)).exp()).log()) / 2
                assert Dup[k] == AT.ceil_scaled(dv, R_ + e_), "Dup differs from Arb"
                if k < len(Dlo):
                    assert Dlo[k] == AT.floor_scaled(dv, R_ + e_), "Dlo differs from Arb"
        for q in (Fraction(1, 7), Fraction(3, 2), Fraction(11, 2)):
            ea = (AT.arb_frac(q)).erf()
            ev = V.erf(q)
            assert AT.lo(ea) <= ev.hi_frac() and ev.lo_frac() <= AT.hi(ea), "erf enclosures disjoint"
        print("1. interval enclosures ok; delta tables identical to Arb (python-flint) tables", flush=True)
    else:
        print("1. interval enclosures ok (python-flint not installed: Arb cross-check skipped)", flush=True)
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
    # 2b. exact CDF of |X+Y| (exact.conv_exact_pieces) against a brute-force rectangle-area computation
    for trial in range(3):
        X = rand_law(rng, 7, trial != 1)
        Y = rand_law(rng, 7, trial != 2)
        Q4, lin, quad = exact.conv_exact_pieces(X, Y)
        for k in range(7):
            for sig in (Fraction(0), Fraction(1, 3), Fraction(4, 5)):
                formula = (Q4[k] + lin[k] * sig + quad[k] * sig * sig) / (4 * ONE * ONE)
                assert formula == brute_abs_sum_cdf(X, Y, k + sig), "conv formula mismatch"
    print("2b. exact |X+Y| CDF formula equals brute-force rectangle areas", flush=True)
    # 3. small-grid laws
    from scipy.special import erf
    e, N, beta = 5, 400, Fraction(5, 4)
    h = 2.0 ** -e
    nu = construct.gauss_law(beta, e, N)
    exact.validate(*nu.pair(), "nu0")
    gauss.check_gauss(*nu.pair(), beta, e)
    t = np.linspace(0, N * h, 20001)
    d = cdf_float(nu, h, t) - erf(t / (1.25 * math.sqrt(2)))
    print(f"3a. gauss: bound - erf in [{d.min():.2e}, {d.max():.2e}]", flush=True)
    assert d.min() > -1e-12
    c = construct.coarsen(nu)
    exact.check_coarsen(nu.pair(), c.pair())
    M2 = construct.min2(nu)
    exact.check_min2(nu.pair(), M2.pair())
    Z = construct.conv(nu, nu)
    exact.check_conv(nu.pair(), nu.pair(), Z.pair())
    U, info = branch.fine_bounds(nu.atom, nu.m, e, 4, 4, 14, 12, kernel="numba")
    Y = construct.branch_from_bounds(U, N, 4)
    exact.check_branch_output(Y.pair(), U, 4)
    # float reference for P(f(A,B) <= u) (quadrature over B)
    C = np.array(nu.cdf(), dtype=float) / ONE
    m = np.array(nu.m, dtype=float) / ONE
    sub = 64
    bmid = (np.arange(N * sub) + 0.5) * h / sub
    wb = np.repeat(m / sub, sub)

    def FA(xx):
        k = np.minimum((xx / h).astype(np.int64), N)
        return np.where(k >= N, 1.0, C[np.minimum(k, N)] + m[np.minimum(k, N - 1)] * (xx / h - k))

    worst = 1.0
    for i in range(0, N * 4, 13):
        u = i * h / 4
        sel = bmid > u
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.arctanh(np.minimum(np.tanh(u) / np.tanh(bmid[sel]), 1.0))
        g = np.where(np.isfinite(g), g, 1e9)
        ref = nu.atom / ONE + wb[~sel].sum() + (wb[sel] * FA(g)).sum()
        worst = min(worst, U[i] / ONE - ref)
    print(f"3b. branch: min(U - float reference) = {worst:.2e} (should be >= -1e-6: quadrature noise)", flush=True)
    assert worst > -1e-6
    print("3. constructor outputs pass all exact checks", flush=True)
    # 4. kernels
    e2, N2 = 7, 1200
    nu2 = construct.gauss_law(Fraction(3, 2), e2, N2)
    for sb in (1, 4):
        U1, _ = branch.fine_bounds(nu2.atom, nu2.m, e2, 4, sb, 14, 12, kernel="numba")
        U2, _ = branch.fine_bounds(nu2.atom, nu2.m, e2, 4, sb, 14, 12, kernel="numpy")
        assert U1 == U2, "kernels differ"
    e3, N3 = 5, 160
    nu3 = construct.gauss_law(Fraction(3, 2), e3, N3)
    for sb in (1, 2):
        U1, _ = branch.fine_bounds(nu3.atom, nu3.m, e3, 4, sb, 14, 12, kernel="numba")
        U3, _ = branch.fine_bounds(nu3.atom, nu3.m, e3, 4, sb, 14, 12, kernel="python")
        assert U1 == U3, "numba and pure-Python kernels differ"
    print("4. numba, numpy and pure-Python kernels agree bit for bit", flush=True)
    # 5. tamper tests: replace a verified output by a strictly more spread law on the same grid
    Z4 = construct.conv(Z, Z)  # law of a 4-fold sum: strictly more spread than |X+Y|
    expect_fail(exact.check_conv, nu.pair(), nu.pair(), Z4.pair())
    expect_fail(exact.check_min2, nu.pair(), construct.conv(M2, M2).pair())
    expect_fail(exact.check_branch_output, construct.conv(Y, Y).pair(), U, 4)
    expect_fail(exact.check_coarsen, nu.pair(), construct.conv(c, c).pair())
    # nu_0 built for beta = 5/4 stays valid for every beta >= 5/4 but must fail for beta = 6/5 (less spread)
    gauss.check_gauss(*nu.pair(), Fraction(13, 10), e)
    expect_fail(gauss.check_gauss, *nu.pair(), Fraction(6, 5), e)
    if a.family:
        from mkcert.family import load_family, verify_family
        fam = load_family(a.family, a.n)
        verify_family(fam)
        for key, factor in (("lam", Fraction(11, 10)), ("eps0", Fraction(10))):
            bad = dict(fam)
            bad[key] = fam[key] * factor
            expect_fail(verify_family, bad)
        print("5b. family tamper tests (lam x1.1, eps0 x10) rejected", flush=True)
    print("5. tamper tests rejected as expected", flush=True)
    print("SELFTEST PASSED", flush=True)


if __name__ == "__main__":
    main()
