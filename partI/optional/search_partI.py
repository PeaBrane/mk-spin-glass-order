#!/usr/bin/env python3
"""SEARCH (optional, not part of the proof): find Part I certificate parameters with floats and write frozen inputs.

Needs scipy (Nelder-Mead).  Every written input is then verified by the standard-library checker (propa.check);
nothing found here is trusted.  The search follows the one that produced the frozen inputs: per level, minimise
the float Gamma(r, w, L, R) over (log L, log R); at every level try to close the tail (Lemma T) with the smallest
admissible A; keep the level that gives the best (order, imbalance) margin.

  python3 partI/optional/search_partI.py cert 4 712.4 --claims bounds --out out/n4_b712.4.json
  python3 partI/optional/search_partI.py pole 8 57.8 --out out/n8_pole.json
  python3 partI/optional/search_partI.py thresh 8 1 2000 0.1 --claim order --out out/n8_order.json
"""
import argparse
import json
import math
import os
import sys
from fractions import Fraction as Fr
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import propa  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

PHI_F = 1 / math.sqrt(2 * math.pi)
LN2 = math.log(2)


def dec(x):
    return f"{x:.12g}"


class Model:
    def __init__(self, n):
        self.n = n
        self.kapf = [None] + [float(propa.kappa(g)) for g in range(1, n + 1)]

    def gp_f(self, r, w, L, R):
        n = self.n
        rho = r / (1 - w)
        p = 2 * w + (2 * rho * R) ** 2 + 4 * rho * L
        th = (1 + math.exp(-2 * L)) / math.tanh(R)
        if not (p < 1 and w < 1):
            return 1e300, max(p, 1.0)
        return th * sum(comb(n, g) * self.kapf[g] * p ** (n - g) for g in range(1, n + 1)), p

    def best(self, r, w, x0=None):
        def obj(x):
            if max(abs(x[0]), abs(x[1])) > 300:
                return 1e300
            return self.gp_f(r, w, math.exp(x[0]), math.exp(x[1]))[0]
        starts = ([x0] if x0 is not None else []) + [[math.log(1.5), math.log(3.0)], [math.log(3.0), math.log(5.0)],
                                                     [math.log(0.05), math.log(2.0)]]
        bestv = None
        for st in starts:
            res = minimize(obj, st, method="Nelder-Mead", options=dict(xatol=1e-11, fatol=1e-16, maxiter=4000))
            if bestv is None or res.fun < bestv.fun:
                bestv = res
        L, R = float(dec(math.exp(bestv.x[0]))), float(dec(math.exp(bestv.x[1])))
        G, p = self.gp_f(r, w, L, R)
        return L, R, G, p, list(bestv.x)


def tail_try(model, r, w):
    n = model.n
    A = max(1.000001 * w / r ** 2, 1e-250)
    for _ in range(12):
        if A * r * r >= 1:
            return None
        L, R, G, p, _ = model.best(r, A * r * r)
        if G >= 1:
            return None
        A_new = max(1.000001 * w / r ** 2, 1.01 * p ** n / (G * G * r * r), 1e-250)
        if A_new <= A:
            break
        A = A_new
    A = float(dec(A * 1.0000001))
    if A * r * r >= 1:
        return None
    L, R, G, p, _ = model.best(r, A * r * r)
    gamma = float(dec(G * (1 + 1e-10) + 1e-14))
    if gamma >= 1 or p ** n > 0.999 * A * gamma ** 2 * r ** 2:
        return None
    return dict(A=dec(A), gamma=dec(gamma), L=dec(L), R=dec(R))


def ell_est(r, w, t, ls, lb, S, P):
    A, g = float(t["A"]), float(t["gamma"])
    ls2 = ls + A * r * r / (2 * (1 - g * g)) + 2 * r / (1 - g)
    lb2 = lb + (0.75 * A * r * r + A * A * r ** 4 + A * r ** 3 + r * r / 2) / (1 - g * g) + (1 + 2 * LN2) * r / (1 - g)
    return ls2, lb2, max(S, A * r * r + 2 * r), max(P, A * r * r + 2 * LN2 * r)


def search(model, beta_str, pole_only, kmax=3000):
    r, w = PHI_F / float(Fr(beta_str)), 0.0
    if pole_only:
        t = tail_try(model, r, w)
        return None if t is None else ([], t)
    levels, x0 = [], None
    ls = lb = S = P = 0.0
    best = None
    for k in range(kmax):
        t = tail_try(model, r, w)
        if t is not None:
            e = ell_est(r, w, t, ls, lb, S, P)
            score = (1 - 2 * e[0] - e[2], (1 - 2 * e[1]) ** 2 - e[3])
            if best is None or score > best[0]:
                best = (score, list(levels), t)
            if 2 * r / (1 - float(t["gamma"])) < 1e-7 * max(ls, 1e-12):
                break
        S, P = max(S, w + 2 * r), max(P, w + 2 * LN2 * r)
        ls += w / 2 + 2 * r
        lb += 0.75 * w + (1 + 2 * LN2) * r + w * w + w * r + r * r / 2
        L, R, G, p, x0 = model.best(r, w, x0)
        levels.append([dec(L), dec(R)])
        r, w = r * G, p ** model.n
        if w >= 0.5 or r > 1:
            break
    return None if best is None else (best[1], best[2])


def certificate(n, beta_str, claims, model=None):
    """Search, then verify with the trusted standard-library checker; returns the input dict or None."""
    model = model or Model(n)
    found = search(model, beta_str, pole_only=claims == ["pole"])
    if found is None:
        return None
    levels, tail = found
    cert = {"kind": "mk-partI-certificate", "n": n, "couplings": "gaussian", "beta": beta_str, "claims": claims,
            "levels": levels, "tail": tail,
            "description": "found by partI/optional/search_partI.py (float Nelder-Mead; untrusted)"}
    try:
        propa.check(cert)
    except propa.CertificateError:
        return None
    return cert


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cert")
    c.add_argument("n", type=int)
    c.add_argument("beta")
    c.add_argument("--claims", default="bounds")
    c.add_argument("--out", required=True)
    p = sub.add_parser("pole")
    p.add_argument("n", type=int)
    p.add_argument("beta")
    p.add_argument("--out", required=True)
    t = sub.add_parser("thresh")
    t.add_argument("n", type=int)
    t.add_argument("lo")
    t.add_argument("hi")
    t.add_argument("step")
    t.add_argument("--claim", choices=["pole", "order", "imbalance"], required=True)
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd in ("cert", "pole"):
        claims = ["pole"] if a.cmd == "pole" else a.claims.split(",")
        cert = certificate(a.n, a.beta, claims)
    else:
        model = Model(a.n)
        st = Fr(a.step)
        lo_m, hi_m = int(Fr(a.lo) / st), int(Fr(a.hi) / st)

        def bstr(m):
            v = Fr(m) * st
            return str(v.numerator) if v.denominator == 1 else f"{float(v):.10g}"

        def ok(m):
            return certificate(a.n, bstr(m), [a.claim], model)

        top = ok(hi_m)
        if top is None:
            sys.exit(f"hi = {bstr(hi_m)} does not certify")
        if ok(lo_m) is not None:
            sys.exit(f"lo = {bstr(lo_m)} already certifies")
        while hi_m - lo_m > 1:
            mid = (lo_m + hi_m) // 2
            got = ok(mid)
            if got is not None:
                hi_m, top = mid, got
            else:
                lo_m = mid
        cert = top
        cert["description"] += f"; the search did not certify beta = {bstr(lo_m)}"
        print(f"n = {a.n}, claim {a.claim}: certified beta = {bstr(hi_m)}, search failed at {bstr(lo_m)}")
    if cert is None:
        sys.exit("no certificate found")
    with open(a.out, "w") as fh:
        json.dump(cert, fh, indent=1)
        fh.write("\n")
    print(f"wrote {a.out} (verified by the standard-library checker)")


if __name__ == "__main__":
    main()
