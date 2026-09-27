#!/usr/bin/env python3
"""CHECK: re-verify one certificate point of the computer-assisted theorem from its frozen inputs.

Usage:
  python3 check.py --n 4 --point ../inputs/point_n4.json --threads 1 --kernel python --out ../runs/check_n4.json
  (standard library only; --kernel numpy or numba evaluates the same integer sums faster)

Everything that defines the certificate is read from the frozen point file and the family file it names (whose
SHA-256 must match).  `--n` must equal the n recorded in both files.  `--threads` and `--kernel` only choose how
the integer sums are evaluated and do not change any number.  The output records n, every parameter, the
SHA-256 of every input and code file, all verified margins, the Theorem 2/3 constants and the cost.

Checked (see CONDITIONS.md for the condition-to-code map):
  family  (F1)-(F4): validity, G_1 >= min2(X_c), G_k >= conv(G_{k-1}, G_1), (C1), (C2);
  bridge  (B0) nu_0 >= law of beta|J|; per level: (B1) coarsening, (B2) branch output >= rigorous fine bounds,
          (B3) the n-fold plan labels and every conv output >= exact CDF; (B4) nu_J enters the family:
          CDF <= alpha0 + (1-alpha0) F_c(./L0);
  constants: exact rational lower bounds of kappa_Q, kappa_C from nu_0..nu_J and the family tail.
"""
import argparse
import json
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mkcert import bridge, constants, exact  # noqa: E402
from mkcert import ivdec as V  # noqa: E402
from mkcert.common import as_int, code_hashes, dump_json, environment, frac, load_json, peak_rss_mb, sha256_file  # noqa: E402
from mkcert.family import load_family, verify_family  # noqa: E402


def log10_floor(x, decimals=3):
    """Decimal string of the largest multiple of 10^-decimals that is <= log10(x) (x a positive Fraction);
    decided on the lower endpoint of an interval enclosure."""
    v = (V.log(V.IV.of(x)) / V.log(V.IV.of(10))).lo_frac() * 10 ** decimals
    k = v.numerator // v.denominator
    sign = "-" if k < 0 else ""
    q, r = divmod(abs(k), 10 ** decimals)
    return f"{sign}{q}.{r:0{decimals}d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="number of branches (must match the inputs)")
    ap.add_argument("--point", required=True, help="frozen point file (inputs/point_n*.json)")
    ap.add_argument("--threads", type=int, required=True, help="numba threads (does not affect results)")
    ap.add_argument("--kernel", choices=["numba", "numpy", "python"], required=True,
                    help="integer-sum kernel (all three compute identical integers)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    if a.kernel == "numba":
        import numba
        numba.set_num_threads(a.threads)
    point_path = os.path.abspath(a.point)
    pt = load_json(point_path)
    exact.need(pt.get("kind") == "mk-bridge-certificate", "point: wrong kind")
    exact.need(as_int(pt["n"], "point n") == a.n, f"point file is for n = {pt['n']}, requested n = {a.n}")
    exact.need(isinstance(pt["family"]["file"], str), "point: family.file must be a string")
    fam_path = os.path.join(os.path.dirname(point_path), pt["family"]["file"])
    fam_sha = sha256_file(fam_path)
    exact.need(fam_sha == pt["family"]["sha256"], "family file SHA-256 does not match the point file")
    fam = load_family(fam_path, a.n)
    beta = frac(pt["beta"], "beta")
    g, br = pt["grid"], pt["branch"]
    e0, N = as_int(g["e0"], "e0"), as_int(g["N"], "N")
    s, sb, R, dmax = (as_int(br[k], k) for k in ("s", "sb", "R", "dmax"))
    exact.need(isinstance(pt["coarsen"], list), "point: coarsen must be a JSON list")
    schedule = [as_int(c, "coarsen entry") for c in pt["coarsen"]]
    J = as_int(pt["J"], "J")
    exact.need(beta > 0 and N >= 2 and N % 2 == 0 and J >= 0 and min(schedule, default=0) >= 0,
               "point: parameters out of range")
    exact.need("T" not in pt or frac(pt["T"], "T") == 1 / beta, "point: T != 1/beta")
    out = {
        "kind": "mk-certificate-check",
        "n": a.n, "beta": str(beta), "T": str(1 / beta),
        "parameters": {"e0": e0, "N": N, "s": s, "sb": sb, "R": R, "dmax": dmax, "J": J, "coarsen": schedule,
                       "lam": str(fam["lam"]), "A": str(fam["A"]), "eps0": str(fam["eps0"]),
                       "xmax": str(fam["xmax"]), "Nx": fam["Nx"], "Npad": fam["Npad"], "P": exact.P,
                       "interval_digits": V.PREC, "constants_Q": constants.Q},
        "inputs": {os.path.basename(point_path): sha256_file(point_path), os.path.basename(fam_path): fam_sha},
        "code": code_hashes(),
        "execution": {"threads": a.threads, "kernel": a.kernel,
                      "environment": environment(mods=("numpy", "numba") if a.kernel != "python" else ())},
        "checks": {},
    }
    # ---------------------------------------------------------------- constants c0
    out["checks"]["rational_constants"] = constants.verify_rational_constants()
    L0 = constants.C0_UP / fam["eps0"]  # L0 >= c0 / eps0
    alpha0 = fam["A"] * constants.C0_LO / L0  # alpha0 <= A c0 / L0
    out["parameters"]["L0"] = str(L0)
    out["parameters"]["alpha0"] = str(alpha0)
    # ---------------------------------------------------------------- family (Theorem 1A)
    tf = time.time()
    fr = verify_family(fam)
    fr["time_s"] = time.time() - tf
    out["checks"]["family"] = {k: (str(v) if isinstance(v, Fraction) else v) for k, v in fr.items()}
    out["checks"]["family"]["C2_min_margin_over_eps0_float"] = float(fr["C2_min_margin_over_eps0"])
    print("family ok", json.dumps(out["checks"]["family"]), flush=True)

    # ---------------------------------------------------------------- bridge (Theorem 1B)
    def log(rec):
        if rec["level"] % 10 == 0 or "entry_margin" in rec:
            print(json.dumps(rec), flush=True)

    res = bridge.run(a.n, beta, e0, N, s, sb, R, dmax, fam, L0, alpha0, mode="check", schedule=schedule, J=J,
                     kernel=a.kernel, log=log)
    exact.need(res["status"] == "CERTIFIED", "bridge did not certify")
    fp = pt.get("fingerprint_chain_sha256")
    out["checks"]["bridge"] = {
        "status": res["status"], "J": res["J"], "e_final": res["e_final"],
        "gauss_worst_upper": res["levels"][0].get("gauss_worst_upper"),
        "branch_min_fine_slack": str(res["worst_branch_slack"]),
        "conv_min_knot_slack": str(res["worst_conv_slack"]),
        "entry_max_lhs_minus_rhs": res["levels"][-1]["entry_margin"],
        "counts": res["counts"],
        "chain_sha256": res["chain_sha256"],
        "chain_matches_search_fingerprint": (fp == res["chain_sha256"]) if fp else None,
        "time_s": res["time_s"],
    }
    out["levels"] = res["levels"]
    # ---------------------------------------------------------------- Theorem 2/3 constants
    tk = time.time()
    a1, a2 = constants.family_tail_coeffs(fam["A"], fam["fc"])
    kp = constants.kappas(a.n, res["phis"], J, a1, a2, L0, fam["lam"])
    kQ, kC = kp["kappa_Q"], kp["kappa_C"]
    exact.need(kQ > 0 and kC > 0, "constants not positive")
    out["constants"] = {
        "S_upper_ceil": exact.ceil_sci(kp["S_upper"]),
        "kappa_Q_floor": exact.floor_sci(kQ), "log10_kappa_Q_floor": log10_floor(kQ),
        "kappa_C_floor": exact.floor_sci(kC), "log10_kappa_C_floor": log10_floor(kC),
        "EQ2_lower_floor": exact.floor_sci(kQ * kQ), "log10_EQ2_lower_floor": log10_floor(kQ * kQ),
        "imbalance_lower_floor": exact.floor_sci(kC * kC),
        "log10_imbalance_lower_floor": log10_floor(kC * kC),
        "j_first_T_below_half": kp["j_first_T_below_half"], "j_first_T_below_one": kp["j_first_T_below_one"],
        "j_max_summed": kp["j_max"],
        "T_upper_first10": [exact.ceil_sci(kp["T_upper"][j]) for j in range(1, min(J + 1, 10) + 1)],
        "phi_upper_first10": [exact.ceil_sci(p) if p > 0 else "0" for p in res["phis"][:10]],
        "family_tail_coeffs_upper": {"a1": exact.ceil_sci(a1), "a2": exact.ceil_sci(a2)},
        "f_c": str(fam["fc"]),
        "time_s": time.time() - tk,
    }
    out["verdict"] = "ALL CHECKS PASSED"
    out["cost"] = {"wall_s": time.time() - t0, "peak_rss_MB": peak_rss_mb()}
    dump_json(out, a.out)
    print(json.dumps({k: out[k] for k in ("n", "beta", "verdict", "constants", "cost")}, indent=1), flush=True)


if __name__ == "__main__":
    try:
        main()
    except exact.CertificateError as ex:
        print("CHECK FAILED:", ex, flush=True)
        sys.exit(2)
