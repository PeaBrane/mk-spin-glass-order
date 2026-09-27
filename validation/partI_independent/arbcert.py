"""Second implementation of the Part I check in python-flint Arb ball arithmetic (independent of qcert's series).

Same mathematics: H_0(phi/beta, 0); Prop A per level; Lemma T at level K; Transfer-Theorem constants.
Carried r_k, w_k are replaced by the upper endpoints of their balls (valid H parameters by monotonicity).
Every decision requires the ball comparison to be decided; undecided counts as failure.
"""

import json
import math
import sys
from fractions import Fraction

from flint import arb, ctx, fmpq

ctx.prec = 400


def A_(s):
    q = Fraction(s)
    return arb(fmpq(q.numerator, q.denominator))


def kappa(g):
    m = 2 * g
    num = 2 * sum((-1) ** j * math.comb(m, j) * (g - j) ** (m - 1) for j in range(g + 1))
    return fmpq(num, math.factorial(m - 1))


def up(x):
    return arb(x.upper())


def check(cert):
    n = cert["n"]
    beta = A_(cert["beta"])
    coef = [arb(math.comb(n, n - j) * kappa(n - j)) for j in range(n)]
    ln2 = arb(2).log()
    r = up(1 / (2 * arb.pi()).sqrt() / beta)
    w = arb(0)
    rs, ws = [r], [w]
    fails = []

    def prop_a(r, w, L, R):
        if not (w < 1):
            fails.append("w<1")
        if not (L > 0 and R > 0):
            fails.append("L,R>0")
        rho = r / (1 - w)
        p = 2 * w + (2 * rho * R) ** 2 + 4 * rho * L
        theta = (1 + (-2 * L).exp()) / R.tanh()
        acc = coef[n - 1]
        for j in range(n - 2, -1, -1):
            acc = acc * p + coef[j]
        return theta * acc, p

    for Ls, Rs in cert["levels"]:
        G, p = prop_a(r, w, A_(Ls), A_(Rs))
        r, w = up(r * G), up(p ** n)
        rs.append(r)
        ws.append(w)
    t = cert["tail"]
    Acoef, gam, L, R = A_(t["A"]), A_(t["gamma"]), A_(t["L"]), A_(t["R"])
    rK, wK = rs[-1], ws[-1]
    W = Acoef * rK * rK
    hyp = {"T0a": bool(wK <= W), "T0b": bool(W < 1)}
    G, p = prop_a(rK, W, L, R)
    hyp["T1_Gamma_le_gamma"] = bool(G <= gam)
    hyp["T1_gamma_lt_1"] = bool(gam < 1)
    hyp["T2"] = bool(p ** n <= Acoef * gam * gam * rK * rK)
    closes = all(hyp.values()) and not fails
    cb = 1 + 2 * ln2
    ls = arb(0)
    lb = arb(0)
    S = []
    Pc = []
    for rk, wk in zip(rs, ws):
        ls += wk / 2 + 2 * rk
        lb += arb(3) / 4 * wk + cb * rk + wk * wk + wk * rk + rk * rk / 2
        S.append(wk + 2 * rk)
        Pc.append(wk + 2 * ln2 * rk)

    def geo(j):
        gj = gam ** j
        return gj / (1 - gj)

    ls += W / 2 * geo(2) + 2 * rK * geo(1)
    lb += arb(3) / 4 * W * geo(2) + cb * rK * geo(1) + W * W * geo(4) + W * rK * geo(3) + rK * rK / 2 * geo(2)
    S.append(W * gam * gam + 2 * rK * gam)
    Pc.append(W * gam * gam + 2 * ln2 * rK * gam)
    # conservative: S and P are the largest upper endpoints
    Su = arb(max((s.upper() for s in S), key=lambda a: float(a)))
    Pu = arb(max((s.upper() for s in Pc), key=lambda a: float(a)))
    om = 1 - 2 * ls - Su
    one_m = 1 - 2 * lb
    one_m_plus = one_m if bool(one_m > 0) else arb(0)
    im = one_m_plus * one_m_plus - Pu
    verdict = {}
    for c in cert["claims"]:
        if c in ("pole", "bounds"):
            verdict[c] = closes
        elif c == "order":
            verdict[c] = closes and bool(om > 0)
        elif c == "imbalance":
            verdict[c] = closes and bool(one_m > 0) and bool(im > 0)
    return {"n": n, "beta": cert["beta"], "K": len(cert["levels"]), "hyp": hyp, "level_fails": sorted(set(fails)),
            "verdict": verdict, "all": all(verdict.values()),
            "order_margin_lo": float(om.lower()), "imb_margin_lo": float(im.lower()),
            "pole_lo": float((1 - Su).lower()), "EA_lo": float((1 - 2 * ls).lower()),
            "T1_rel": float(((gam - G) / gam).lower()), "gamma_minus_1": float((gam - 1).upper()),
            "T2_ratio": float((p ** n / (Acoef * gam * gam * rK * rK)).upper()),
            "T0_ratio": float((wK / W).upper()) if float(W.upper()) > 0 else None}


if __name__ == "__main__":
    for path in sys.argv[1:]:
        with open(path) as fh:
            cert = json.load(fh)
        res = check(cert)
        res["file"] = path
        print(json.dumps(res), flush=True)
