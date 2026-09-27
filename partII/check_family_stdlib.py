#!/usr/bin/env python3
"""Standard-library-only verifier of a Theorem 1A family certificate.

  python3 check_family_stdlib.py --n 4 --family ../inputs/family_n4.json --out ../runs/family_n4_stdlib.json

Uses only the Python standard library (integers, fractions, hashlib, json): it imports mkcert.exact,
mkcert.family and mkcert.common, none of which imports numpy, numba or python-flint.  It checks (F1)-(F4):
validity of X_c and G_1..G_n, G_1 >= min2(X_c), G_k >= conv(G_{k-1}, G_1), and conditions (C1), (C2), all in
exact integer / rational arithmetic.  The output records n, all parameters and the SHA-256 of the input and code.
"""
import argparse
import json
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mkcert import exact  # noqa: E402
from mkcert.common import dump_json, environment, peak_rss_mb, sha256_file  # noqa: E402
from mkcert.family import load_family, verify_family  # noqa: E402

STDLIB_FILES = ["check_family_stdlib.py", "mkcert/__init__.py", "mkcert/exact.py", "mkcert/family.py",
                "mkcert/common.py"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    fam = load_family(os.path.abspath(a.family), a.n)
    res = verify_family(fam)
    here = os.path.dirname(os.path.abspath(__file__))
    heavy = sorted(m for m in ("numpy", "numba", "flint") if m in sys.modules)
    out = {
        "kind": "mk-family-check-stdlib", "n": a.n,
        "parameters": {k: str(fam[k]) for k in ("lam", "A", "eps0", "xmax", "Nx", "Npad", "hc")},
        "inputs": {os.path.basename(a.family): sha256_file(a.family)},
        "code": {f: sha256_file(os.path.join(here, f)) for f in STDLIB_FILES},
        "result": {k: (str(v) if isinstance(v, Fraction) else v) for k, v in res.items()},
        "C2_min_margin_over_eps0_float": float(res["C2_min_margin_over_eps0"]),
        "verdict": "ALL FAMILY CHECKS PASSED",
        "non_stdlib_modules_loaded": heavy,
        "environment": environment(mods=()),
        "cost": {"wall_s": time.time() - t0, "peak_rss_MB": peak_rss_mb()},
    }
    dump_json(out, a.out)
    print(json.dumps({k: out[k] for k in ("n", "verdict", "C2_min_margin_over_eps0_float",
                                           "non_stdlib_modules_loaded", "cost")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except exact.CertificateError as ex:
        print("CHECK FAILED:", ex, flush=True)
        sys.exit(2)
