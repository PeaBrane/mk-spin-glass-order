#!/usr/bin/env python3
"""Validation of the Part I checker (standard library only; NOT part of the proof).

  python3 partI/selftest_partI.py

1. kappa_g: the Irwin-Hall formula equals an independent Eulerian-number formula for g <= 64; kappa_1..4 are
   2, 4/3, 11/10, 302/315; kappa_g is nonincreasing and kappa_3 > 1 > kappa_4.
2. round_up: x <= round_up(x) <= x (1 + 10^(1-DIGITS)) on random rationals.
3. Prop A Horner sum (rounded up at every step) >= the exact sum and agrees with it to 10^-60 relative.
4. The transcendental bounds used (phi = 1/sqrt(2 pi), ln 2, e^{-x}) overlap independent rational enclosures
   built here from Taylor partial sums and a decimal value of pi (a second implementation, not sharing ivdec).
5. Gamma at the nine hand-checkable one-shot rows agrees (to 1e-9) with the values computed by an independent
   exact-rational implementation.
6. The multi-level bounds at n = 4, beta = 712.4 equal the 6-decimal values of the original Arb (python-flint,
   256-bit) checker: EA_fixed 0.891741, E Q^2 0.793206, E D^2 0.757554.
"""
import os
import random
import sys
from fractions import Fraction
from math import comb, factorial, isqrt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import propa  # noqa: E402
from propa import kappa, round_up  # noqa: E402


def eulerian(m, k):
    """Eulerian number A(m, k) (permutations of m with k descents)."""
    return sum((-1) ** j * comb(m + 1, j) * (k + 1 - j) ** m for j in range(k + 2))


def exp_bounds(x, terms=200):
    """Rational [lo, hi] with lo <= e^x <= hi for rational 0 <= x <= 40 (Taylor; remainder by a geometric tail)."""
    s, t = Fraction(0), Fraction(1)
    for k in range(terms):
        s += t
        t = t * x / (k + 1)
    # remaining terms t_terms, t_terms+1, ... with ratio <= x / (terms + 1) < 1
    q = x / (terms + 1)
    assert q < 1
    return s, s + t / (1 - q)


def main():
    # 1. kappa
    assert [kappa(g) for g in range(1, 5)] == [2, Fraction(4, 3), Fraction(11, 10), Fraction(302, 315)]
    for g in range(1, 65):
        # Irwin-Hall density at the integer g of a sum of 2g uniforms: A(2g-1, g-1) / (2g-1)!
        assert kappa(g) == Fraction(2 * eulerian(2 * g - 1, g - 1), factorial(2 * g - 1)), g
        if g > 1:
            assert kappa(g) <= kappa(g - 1)
    assert kappa(3) > 1 > kappa(4)
    print("1. kappa_g: Irwin-Hall = Eulerian formula for g <= 64; kappa_1..4 = 2, 4/3, 11/10, 302/315; monotone")
    # 2. round_up
    rng = random.Random(7)
    for _ in range(2000):
        x = Fraction(rng.randrange(1, 10 ** rng.randrange(1, 90)), rng.randrange(1, 10 ** rng.randrange(1, 90)))
        y = round_up(x)
        assert x <= y <= x * (1 + Fraction(1, 10 ** (propa.DIGITS - 1))) and round_up(y) == y
    print("2. round_up: upward, relative error <= 10^-63, idempotent (2000 random rationals)")
    # 3. Horner with upward rounding versus the exact sum
    for n in (4, 7, 16, 64):
        kap = propa.kappas(n)
        for _ in range(5):
            r = Fraction(rng.randrange(1, 10 ** 6), 10 ** 9)
            w = Fraction(rng.randrange(0, 10 ** 6), 10 ** 12)
            L, R = Fraction(rng.randrange(1, 10 ** 4), 10 ** 3), Fraction(rng.randrange(1, 10 ** 4), 10 ** 3)
            gam, p = propa.prop_a(n, kap, r, w, L, R)
            rho = r / (1 - w)
            p_ex = 2 * w + (2 * rho * R) ** 2 + 4 * rho * L
            poly_ex = sum(comb(n, g) * kap[g] * p_ex ** (n - g) for g in range(1, n + 1))
            theta = propa.theta_hi(L, R)
            assert p >= p_ex and gam >= theta * poly_ex
            assert gam <= theta * poly_ex * (1 + Fraction(1, 10 ** 60)), (n, float(gam / (theta * poly_ex) - 1))
    print("3. Prop A Horner sum >= exact sum, relative excess <= 1e-60 (n = 4, 7, 16, 64)")
    # 4. transcendental constants against an independent implementation
    pi_lo, pi_hi = Fraction(314159265358979, 10 ** 14), Fraction(314159265358980, 10 ** 14)

    def sqrt_bounds(x, d=40):
        s = isqrt(x.numerator * 10 ** (2 * d) // x.denominator)
        return Fraction(s, 10 ** d), Fraction(s + 1, 10 ** d)

    phi_lo = 1 / sqrt_bounds(2 * pi_hi)[1]
    phi_hi = 1 / sqrt_bounds(2 * pi_lo)[0]
    assert phi_lo <= propa.PHI_HI <= phi_hi, "phi"
    e_lo, e_hi = exp_bounds(propa.LN2_HI)
    assert e_lo >= 2, "LN2_HI is not an upper bound of ln 2"  # e^{LN2_HI} >= e_lo >= 2
    assert e_hi <= 2 + Fraction(1, 10 ** 60), "LN2_HI is not tight"
    for x in (Fraction(1, 1000), Fraction(5), Fraction(21, 2), Fraction(3676, 100)):
        lo, hi = exp_bounds(x)
        iv = propa.V.exp(propa.V.IV.of(-x))
        # e^{-x} in [1/hi, 1/lo]; the two enclosures must overlap
        assert iv.lo_frac() <= 1 / lo and 1 / hi <= iv.hi_frac(), x
    print("4. phi, ln 2 and e^{-x} bounds overlap independent Taylor / decimal-pi enclosures")
    # 5. Gamma at the one-shot rows versus independent exact-rational values
    ref = {4: "0.9996640918", 5: "0.9975169374", 6: "0.9973091731", 7: "0.9964333524", 8: "0.9944353904",
           12: "0.9925775555", 16: "0.9971600367", 32: "0.9987127265", 64: "0.9968223335"}
    rows = {4: ("525", "5/2", "21/4", "1/100"), 5: ("135", "17/10", "4", "1/1000"), 6: ("86", "7/5", "7/2", "1/10000"),
            7: ("68", "6/5", "16/5", "1/10000"), 8: ("59", "1", "3", "1/10000"), 12: ("44", "3/5", "13/5", "1/10000"),
            16: ("36", "3/10", "7/3", "1/10000"), 32: ("63/5", "1/1000", "5/4", "1/10000"),
            64: ("47/5", "1/1000", "5/6", "1/10000")}
    for n, (b, L, R, A) in rows.items():
        r = round_up(propa.PHI_HI / Fraction(b))
        gam, _ = propa.prop_a(n, propa.kappas(n), r, Fraction(A) * r * r, Fraction(L), Fraction(R))
        assert abs(gam - Fraction(ref[n])) < Fraction(1, 10 ** 9), (n, float(gam))
    print("5. Gamma at the nine one-shot rows agrees with independent exact-rational values to 1e-9")
    # 6. a multi-level certificate against the original Arb checker's printed bounds
    import json
    with open(os.path.join(HERE, "inputs", "bounds", "n4_b712.4.json")) as fh:
        res = propa.check(json.load(fh))
    assert (res["EA_fixed_lower6"], res["EQ2_lower6"], res["ED2_lower6"]) == ("0.891741", "0.793206", "0.757554"), res
    print("6. n = 4, beta = 712.4 (K = 408): EA_fixed, E Q^2, E D^2 bounds equal the Arb checker's to 6 decimals")
    print("PART I SELFTEST PASSED")


if __name__ == "__main__":
    main()
