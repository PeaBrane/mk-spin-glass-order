"""Second, independent checker for Part I certificates (mpmath interval arithmetic, 50 digits).

Shares no code with the float search that produced the certificates. kappa_g is computed from the Irwin-Hall
closed form kappa_g = 2 f_{2g}(g), f_m(x) = sum_{k<=x} (-1)^k C(m,k) (x-k)^{m-1}/(m-1)!  (exact rationals).
Formulas are typed from the statements (Prop A, Lemma T, Transfer-Theorem constants), not from the search code.

usage: check_iv_mpmath.py cert1.json [cert2.json ...]
"""
import json
import math
import sys
from fractions import Fraction as Fr

from mpmath import iv, mpf

iv.dps = 50


def kappa(g):
    m, x = 2 * g, g
    s = sum((-1) ** k * math.comb(m, k) * Fr(x - k) ** (m - 1) for k in range(x + 1))
    return 2 * s / math.factorial(m - 1)


def I(q):
    q = Fr(q)
    return iv.mpf(q.numerator) / q.denominator


def _fr(t):
    sign, man, exp, bc = t
    v = Fr(int(man)) * Fr(2) ** int(exp)
    return -v if sign else v


def hi(x):
    """exact upper endpoint as a Fraction"""
    return _fr(x._mpi_[1])


def lo(x):
    """exact lower endpoint as a Fraction"""
    return _fr(x._mpi_[0])


def upper(x):
    return I(hi(x))


def verify(raw):
    n = int(raw["n"])
    kap = [None] + [I(kappa(g)) for g in range(1, n + 1)]
    one = iv.mpf(1)
    ln2 = iv.log(2)
    beta = I(raw["beta"])
    r = upper(one / iv.sqrt(2 * iv.pi) / beta)
    w = iv.mpf(0)

    def step(r, w, L, R):
        rho = r / (one - w)
        p = 2 * w + (2 * rho * R) ** 2 + 4 * rho * L
        e2R = iv.exp(-2 * R)
        tanhR = (one - e2R) / (one + e2R)
        theta = (one + iv.exp(-2 * L)) / tanhR
        s = iv.mpf(0)
        for g in range(1, n + 1):
            s += math.comb(n, g) * kap[g] * p ** (n - g)
        return theta * s, p

    ls = lb = S = P = iv.mpf(0)
    for (Ls, Rs) in raw["levels"]:
        L, R = I(Ls), I(Rs)
        assert lo(L) > 0 and lo(R) > 0 and hi(w) < 1
        S = I(max(hi(S), hi(w + 2 * r)))
        P = I(max(hi(P), hi(w + 2 * ln2 * r)))
        ls = upper(ls + w / 2 + 2 * r)
        lb = upper(lb + 3 * w / 4 + (1 + 2 * ln2) * r + w * w + w * r + r * r / 2)
        G, p = step(r, w, L, R)
        r, w = upper(r * G), upper(p ** n)
    t = raw["tail"]
    A, gam, LT, RT = I(t["A"]), I(t["gamma"]), I(t["L"]), I(t["R"])
    wT = A * r * r
    GT, pT = step(r, wT, LT, RT)
    ok = (hi(w) <= lo(wT) and hi(wT) < 1 and hi(gam) < 1 and hi(GT) <= lo(gam)
          and hi(pT ** n) <= lo(A * gam ** 2 * r * r) and lo(LT) > 0 and lo(RT) > 0)
    if not ok:
        return dict(ok=False)
    ls = upper(ls + A * r * r / (2 * (one - gam ** 2)) + 2 * r / (one - gam))
    lb = upper(lb + 3 * A * r * r / (4 * (one - gam ** 2)) + (1 + 2 * ln2) * r / (one - gam)
               + (A * A * r ** 4 + A * r ** 3 + r * r / 2) / (one - gam ** 2))
    S = max(hi(S), hi(wT + 2 * r))
    P = max(hi(P), hi(wT + 2 * ln2 * r))
    ms = 1 - 2 * hi(ls) - S
    mb = 1 - 2 * hi(lb)
    # rigorous lower bounds in exact rational arithmetic, floored to 6 decimals
    fl = lambda v: math.floor(v * 10 ** 6) / 10 ** 6
    return dict(ok=True, n=n, beta=raw["beta"], K=len(raw["levels"]), pole=fl(1 - S), EA_fixed=fl(1 - 2 * hi(ls)),
                EQ2=fl(ms ** 2) if ms > 0 else 0.0, ED2=fl((mb ** 2 if mb > 0 else 0) - P),
                ell_s=float(hi(ls)), ell_b=float(hi(lb)))


if __name__ == "__main__":
    for f in sys.argv[1:]:
        raw = json.load(open(f))
        if "levels" not in raw:  # a search record: check every certificate stored in it
            for name, rec in raw["result"].items():
                if "raw" in rec:
                    print(f, name, json.dumps(verify(rec["raw"])))
        else:
            print(f, json.dumps(verify(raw)))
