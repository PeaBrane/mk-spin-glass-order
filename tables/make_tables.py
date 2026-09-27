#!/usr/bin/env python3
"""Write the LaTeX tabular bodies of the paper's tables from checker result files only (standard library).

  python3 tables/make_tables.py --results results --out tables

Reads   RESULTS/partI/*.result.json                 (partI/check_partI.py --manifest ... --out-dir RESULTS/partI)
        RESULTS/partII/check_n{3,4,5,7,8}.json      (partII/check.py ... --out RESULTS/partII/check_n*.json)
Writes  OUT/partI_thresholds.tex   n & beta_pole & beta_order & beta_imb     (0.1 grid, rounded UP to 1 decimal)
        OUT/partI_oneshot.tex      n & beta & L & R & A & gamma              (exact rationals as fractions)
        OUT/partII_points.tex      n & T^* & beta^* & J & log10 kappa_Q & log10 kappa_C  (log10 rounded DOWN, 3 dec.)
        OUT/table_sources.md       every printed number with the result file and field it comes from.
Only rows are written (one per line, ending in \\\\); the manuscript holds the tabular headers.

A result is used only if its verdict is a pass and the SHA-256 it records for its input equals the entry in the
committed input manifest (partI/inputs/SHA256SUMS.txt, partII/inputs/SHA256SUMS.txt); otherwise the script stops.
"""
import argparse
import glob
import json
import os
import sys
from fractions import Fraction

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTI_N = [4, 5, 6, 7, 8, 12, 16, 32, 64]
PARTII_N = [3, 4, 5, 7, 8]


def fail(msg):
    print("make_tables: " + msg, file=sys.stderr)
    sys.exit(2)


def load(path):
    with open(path) as fh:
        return json.load(fh)


def manifest(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                digest, rel = line.split(None, 1)
                out[rel.strip().lstrip("*")] = digest
    return out


def frac_tex(q):
    q = Fraction(q)
    return f"${q.numerator}$" if q.denominator == 1 else f"${q.numerator}/{q.denominator}$"


def ceil_dec1(q):
    """Decimal string with one digit after the point, rounded UP (a certified threshold may only move up)."""
    v = Fraction(q) * 10
    k = -((-v.numerator) // v.denominator)
    return f"{k // 10}.{k % 10}" if k >= 0 else fail("negative threshold")


def check_floor_string(s, decimals):
    """A log10 floor string '-199.009' as recorded by partII/check.py (already rounded DOWN)."""
    whole, dot, frac = s.lstrip("-").partition(".")
    if not (whole.isdigit() and dot and len(frac) == decimals and frac.isdigit()):
        fail(f"unexpected log10 string {s!r}")
    return f"${s}$" if s.startswith("-") else s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src = []  # (table, row, column, value, result file, field)
    rel_res = lambda p: os.path.relpath(p, a.results)  # noqa: E731

    # ------------------------------------------------------------------ Part I
    man1 = manifest(os.path.join(REPO, "partI", "inputs", "SHA256SUMS.txt"))
    recs = {}
    for p in sorted(glob.glob(os.path.join(a.results, "partI", "*.result.json"))):
        d = load(p)
        rel = d["input"]["file"]
        if d.get("verdict") != "PASS":
            fail(f"{p}: verdict is not PASS")
        if man1.get(rel) != d["input"]["sha256"]:
            fail(f"{p}: input {rel} sha256 does not match partI/inputs/SHA256SUMS.txt")
        recs[rel] = (p, d["result"])
    thr = {}
    for rel, (p, r) in recs.items():
        if rel.startswith("thresholds/"):
            for c in r["claims"]:
                key = (r["n"], c)
                if key in thr:
                    fail(f"two threshold certificates for n = {r['n']}, claim {c}")
                thr[key] = (p, r)
    lines = []
    for n in PARTI_N:
        cells = [str(n)]
        for c, col in (("pole", "beta_pole"), ("order", "beta_order"), ("imbalance", "beta_imb")):
            if (n, c) not in thr:
                fail(f"no threshold certificate for n = {n}, claim {c}")
            p, r = thr[(n, c)]
            flag = {"pole": True, "order": r["order_positive"], "imbalance": r["imbalance_positive"]}[c]
            if not flag:
                fail(f"{p}: claim {c} not established")
            v = ceil_dec1(Fraction(r["beta"]))
            cells.append(v)
            src.append(("partI_thresholds", n, col, v, rel_res(p), f"result.beta = {r['beta']} (claim '{c}')"))
        lines.append(" & ".join(cells) + r" \\")
    with open(os.path.join(a.out, "partI_thresholds.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    lines = []
    for n in PARTI_N:
        rel = f"oneshot/oneshot_n{n}.json"
        if rel not in recs:
            fail(f"missing result for {rel}")
        p, r = recs[rel]
        if r["n"] != n or r["K"] != 0 or "pole" not in r["claims"]:
            fail(f"{p}: not a one-shot pole certificate for n = {n}")
        t = r["tail"]
        vals = [("beta", r["beta"]), ("L", t["L"]), ("R", t["R"]), ("A", t["A"]), ("gamma", t["gamma"])]
        cells = [str(n)] + [frac_tex(v) for _, v in vals]
        for (col, v), cell in zip(vals, cells[1:]):
            src.append(("partI_oneshot", n, col, cell, rel_res(p), f"result.{'beta' if col == 'beta' else 'tail.' + col}"))
        lines.append(" & ".join(cells) + r" \\")
    with open(os.path.join(a.out, "partI_oneshot.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ Part II
    man2 = manifest(os.path.join(REPO, "partII", "inputs", "SHA256SUMS.txt"))
    lines = []
    for n in PARTII_N:
        p = os.path.join(a.results, "partII", f"check_n{n}.json")
        if not os.path.exists(p):
            fail(f"missing {p}")
        d = load(p)
        if d.get("verdict") != "ALL CHECKS PASSED" or d.get("n") != n:
            fail(f"{p}: verdict is not ALL CHECKS PASSED for n = {n}")
        for name, digest in d["inputs"].items():
            if man2.get(name) != digest:
                fail(f"{p}: input {name} sha256 does not match partII/inputs/SHA256SUMS.txt")
        b = d["checks"]["bridge"]
        if b["status"] != "CERTIFIED" or b["chain_matches_search_fingerprint"] is not True:
            fail(f"{p}: bridge not certified or chain fingerprint mismatch")
        c = d["constants"]
        T, beta, J = Fraction(d["T"]), Fraction(d["beta"]), d["parameters"]["J"]
        if T != 1 / beta or J != b["J"]:
            fail(f"{p}: inconsistent T, beta or J")
        lq = check_floor_string(c["log10_kappa_Q_floor"], 3)
        lc = check_floor_string(c["log10_kappa_C_floor"], 3)
        cells = [str(n), frac_tex(T), frac_tex(beta), str(J), lq, lc]
        rp = rel_res(p)
        src += [("partII_points", n, "T*", cells[1], rp, "T"), ("partII_points", n, "beta*", cells[2], rp, "beta"),
                ("partII_points", n, "J", cells[3], rp, "parameters.J (= checks.bridge.J)"),
                ("partII_points", n, "log10 kappa_Q", cells[4], rp, "constants.log10_kappa_Q_floor"),
                ("partII_points", n, "log10 kappa_C", cells[5], rp, "constants.log10_kappa_C_floor")]
        lines.append(" & ".join(cells) + r" \\")
    with open(os.path.join(a.out, "partII_points.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    with open(os.path.join(a.out, "table_sources.md"), "w") as fh:
        fh.write("# Sources of the numbers in the table bodies\n\n"
                 f"Generated by `tables/make_tables.py --results {a.results}`. Paths are relative to the results "
                 "directory. Thresholds are rounded up to one decimal; log10 bounds are the checker's floors.\n\n"
                 "| table | n | column | printed | result file | field |\n|---|---|---|---|---|---|\n")
        for t, n, col, v, f, field in src:
            fh.write(f"| {t} | {n} | {col} | {v} | `{f}` | {field} |\n")
    print(f"wrote partI_thresholds.tex, partI_oneshot.tex, partII_points.tex, table_sources.md to {a.out} "
          f"({len(src)} numbers)")


if __name__ == "__main__":
    main()
