#!/usr/bin/env python3
"""SEARCH (optional, not part of the proof): produce the frozen Part II inputs (floats allowed; untrusted).

Needs numpy (shape.py) and, for --kernel numba, numba.  Paths below are relative to partII/optional/.

  python search.py family --n 4 --lam 1133/1000 --A 22/5 --eps0 1/200 --kappa 0.945 --xmax 16/5 --Nx 1600 \
      --Npad 2560 --out ../inputs/family_n4.json
      Builds X_c (float zero-temperature shape, then an exact integer staircase), the witness laws G_1..G_n, and
      runs the exact family verification once (the file is written only if it passes).

  python search.py bridge --n 4 --beta 7/6 --e0 9 --N 12000 --s 4 --sb 4 --R 14 --dmax 12 \
      --family ../inputs/family_n4.json --kernel numba --threads 4 --maxlev 2000 --out ../inputs/point_n4.json
      Runs the bridge, deciding the coarsening schedule heuristically, until the first level J whose law enters
      the family (decided exactly).  Writes the frozen point file: all parameters, the schedule, J, the family
      file's SHA-256 and a fingerprint (SHA-256 of the chain of laws) for reproducibility.

Every certificate parameter is a required argument (no silent defaults).
"""
import argparse
import json
import os
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # shape.py (float construction of X_c)
sys.path.insert(0, os.path.dirname(HERE))  # partII: the mkcert package

import shape  # noqa: E402
from mkcert import bridge, constants, construct, exact  # noqa: E402
from mkcert.common import code_hashes, dump_json, environment, frac, law_to_json, peak_rss_mb, sha256_file  # noqa: E402
from mkcert.family import load_family, verify_family  # noqa: E402


def cmd_family(a):
    t0 = time.time()
    lam, A, eps0, xmax = frac(a.lam, "lam"), frac(a.A, "A"), frac(a.eps0, "eps0"), frac(a.xmax, "xmax")
    x, f, lamstar, its = shape.zero_t_shape(a.n)
    m = shape.discretise(x, f, a.kappa, xmax, a.Nx, a.Npad)
    Xc = construct.Law(0, m)
    G = {1: construct.min2(Xc)}
    for k in range(2, a.n + 1):
        G[k] = construct.conv(G[k - 1], G[1])
    obj = {
        "kind": "mk-family-certificate", "n": a.n, "P": exact.P,
        "lam": str(lam), "A": str(A), "eps0": str(eps0), "xmax": str(xmax), "Nx": a.Nx, "Npad": a.Npad,
        "Xc": law_to_json(*Xc.pair()),
        "G": {str(k): law_to_json(*G[k].pair()) for k in G},
        "search_info": {"shape": "zero-temperature fixed shape (float), scaled by kappa, conditioned on [0,xmax]",
                        "kappa": a.kappa, "lam_star_float": lamstar, "shape_iterations": its,
                        "code": code_hashes(), "environment": environment()},
    }
    fam = load_family(obj, a.n)
    res = verify_family(fam)
    print(json.dumps({k: (float(v) if isinstance(v, Fraction) else v) for k, v in res.items()}), flush=True)
    obj["search_info"]["verified_C2_min_margin_over_eps0"] = float(res["C2_min_margin_over_eps0"])
    obj["search_info"]["time_s"] = time.time() - t0
    dump_json(obj, a.out)
    print(f"wrote {a.out} ({time.time() - t0:.1f}s)", flush=True)


def cmd_bridge(a):
    t0 = time.time()
    if a.kernel == "numba":
        import numba
        numba.set_num_threads(a.threads)
    fam_path = os.path.abspath(a.family)
    fam = load_family(fam_path, a.n)
    constants.verify_rational_constants()
    L0 = constants.C0_UP / fam["eps0"]
    alpha0 = fam["A"] * constants.C0_LO / L0
    beta = frac(a.beta, "beta")
    logf = open(a.log, "w") if a.log else None

    def log(rec):
        line = json.dumps(rec)
        print(line, flush=True)
        if logf:
            print(line, file=logf, flush=True)

    rule = bridge.default_coarsen_rule(a.N)
    res = bridge.run(a.n, beta, a.e0, a.N, a.s, a.sb, a.R, a.dmax, fam, L0, alpha0, mode="search",
                     kernel=a.kernel, maxlev=a.maxlev, log=log, coarsen_rule=rule)
    summary = {k: res[k] for k in ("status", "J", "levels_run", "e_final", "time_s", "chain_sha256")}
    summary["peak_rss_MB"] = peak_rss_mb()
    log(summary)
    if res["status"] != "CERTIFIED":
        print("NOT CERTIFIED; no point file written", flush=True)
        sys.exit(3)
    obj = {
        "kind": "mk-bridge-certificate", "n": a.n, "beta": str(beta), "T": str(1 / beta),
        "grid": {"e0": a.e0, "N": a.N},
        "branch": {"s": a.s, "sb": a.sb, "R": a.R, "dmax": a.dmax},
        "family": {"file": os.path.relpath(fam_path, os.path.dirname(os.path.abspath(a.out))),
                   "sha256": sha256_file(fam_path)},
        "J": res["J"], "coarsen": res["schedule"],
        "fingerprint_chain_sha256": res["chain_sha256"],
        "search_info": {"time_s": time.time() - t0, "peak_rss_MB": peak_rss_mb(), "threads": a.threads,
                        "kernel": a.kernel,
                        "coarsen_rule": "median bin > 0.1 N or mass beyond 0.75 N > 2^-30",
                        "code": code_hashes(), "environment": environment()},
    }
    dump_json(obj, a.out)
    print(f"wrote {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("family")
    for name in ("--lam", "--A", "--eps0", "--xmax", "--out"):
        f.add_argument(name, required=True)
    f.add_argument("--n", type=int, required=True)
    f.add_argument("--kappa", type=float, required=True)
    f.add_argument("--Nx", type=int, required=True)
    f.add_argument("--Npad", type=int, required=True)
    b = sub.add_parser("bridge")
    for name in ("--beta", "--family", "--out"):
        b.add_argument(name, required=True)
    for name in ("--n", "--e0", "--N", "--s", "--sb", "--R", "--dmax", "--threads", "--maxlev"):
        b.add_argument(name, type=int, required=True)
    b.add_argument("--kernel", choices=["numba", "numpy", "python"], required=True)
    b.add_argument("--log", default=None)
    a = ap.parse_args()
    {"family": cmd_family, "bridge": cmd_bridge}[a.cmd](a)


if __name__ == "__main__":
    main()
