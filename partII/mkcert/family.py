"""Theorem 1A family certificate: loading and complete exact verification (standard library only).

A family file fixes n, lam, A, eps0, the grid (x_max, Nx, Npad), the shape X_c and witness laws G_1..G_n.
`verify_family` checks, exactly:
  (F1) X_c and every G_k are valid staircase laws; X_c is atomless with support in [0, x_max];
  (F2) G_1 >= CDF of min(X_1, X_2)                       (exact.check_min2);
  (F3) G_k >= CDF of |W_{k-1} + W_1| for k = 2..n, W_j ~ G_j symmetric (exact.check_conv);
       by Lemma M (Anderson/Birnbaum) G_k then bounds |sum_{i<=k} e_i min(X_i1, X_i2)|;
  (F4) conditions (C1) and (C2) of Theorem 1A, lam > 1, A, eps0 > 0 and 2 A eps0 <= 1   (exact.check_family).
Integers in the file must be JSON integers and each law's recorded sha256 must match its content (common.py).
"""
from fractions import Fraction

from . import exact
from .common import as_int, frac, law_from_json, need


def load_family(path_or_obj, n):
    from .common import load_json
    d = load_json(path_or_obj) if isinstance(path_or_obj, str) else path_or_obj
    need(d.get("kind") == "mk-family-certificate", "family: wrong kind")
    need(as_int(d["n"], "family n") == n, f"family: file is for n = {d['n']}, requested n = {n}")
    need(as_int(d["P"], "family P") == exact.P, "family: unit mismatch")
    fam = {
        "n": n,
        "lam": frac(d["lam"], "lam"),
        "A": frac(d["A"], "A"),
        "eps0": frac(d["eps0"], "eps0"),
        "xmax": frac(d["xmax"], "xmax"),
        "Nx": as_int(d["Nx"], "Nx"),
        "Npad": as_int(d["Npad"], "Npad"),
        "Xc": law_from_json(d["Xc"], "X_c"),
        "G": {k: law_from_json(d["G"][str(k)], f"G_{k}") for k in range(1, n + 1)},
    }
    need(set(d["G"]) == {str(k) for k in range(1, n + 1)}, "family: G must hold exactly G_1..G_n")
    fam["hc"] = fam["xmax"] / fam["Nx"]
    need(len(fam["Xc"][1]) == fam["Npad"], "family: X_c length != Npad")
    for k in range(1, n + 1):
        need(len(fam["G"][k][1]) == fam["Npad"], f"family: G_{k} length != Npad")
    xm = fam["Xc"][1]
    fam["fc"] = Fraction(xm[0], exact.ONE) / fam["hc"]  # max density of X_c: F_c(v) <= fc v
    return fam


def verify_family(fam):
    n = fam["n"]
    Xc, G = fam["Xc"], fam["G"]
    exact.validate(*Xc, "X_c")
    for k in range(1, n + 1):
        exact.validate(*G[k], f"G_{k}")
    s_min2 = exact.check_min2(Xc, G[1])
    s_conv = []
    for k in range(2, n + 1):
        s_conv.append(exact.check_conv(G[k - 1], G[1], G[k]))
    res = exact.check_family(n, fam["lam"], fam["A"], fam["eps0"], fam["hc"], fam["Nx"], Xc, G)
    return {
        "F1_valid_laws": True,
        "F2_min2_min_knot_slack": s_min2,
        "F3_conv_min_knot_slack": min(s_conv) if s_conv else None,
        "C1_min_s": res["C1_min_s"],
        "C2_min_margin": res["C2_min_margin"],
        "C2_min_margin_over_eps0": res["C2_min_margin_over_eps0"],
        "C2_argmin_t": res["C2_argmin_t"],
        "n_intervals": res["n_intervals"],
        "f_c": fam["fc"],
    }
