#!/usr/bin/env python3
"""Compare a fresh results directory with the reference results committed in the repository (stdlib only).

  python3 tools/compare_results.py --ref results --new out

Every certified quantity is compared exactly (as recorded strings or integers): for Part I the whole result record;
for Part II the verdict, parameters, family margins, bridge margins and counts, the chain fingerprint and all
constants.  Timings, environment and peak memory are ignored.  Exit status 2 on any difference.
"""
import argparse
import glob
import json
import os
import sys

IGNORE = {"time_s", "wall_s", "peak_rss_MB", "environment", "execution", "cost", "levels", "t_branch_bounds",
          "t_conv", "t_level"}


def load(p):
    with open(p) as fh:
        return json.load(fh)


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in IGNORE}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'new' if k in b else 'ref'}")
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
    elif a != b:
        out.append(f"{path}: ref {str(a)[:80]!r} != new {str(b)[:80]!r}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--ignore-code", action="store_true", help="also ignore recorded code hashes")
    a = ap.parse_args()
    if a.ignore_code:
        IGNORE.add("code")
    pairs = []
    for sub, pat in (("partI", "*.result.json"), ("partII", "check_n*.json"), ("partII", "family_n*.json")):
        for p in sorted(glob.glob(os.path.join(a.ref, sub, pat))):
            pairs.append((p, os.path.join(a.new, sub, os.path.basename(p))))
    if not pairs:
        print(f"no reference results under {a.ref}")
        sys.exit(2)
    bad = 0
    for ref, new in pairs:
        name = os.path.relpath(ref, a.ref)
        if not os.path.exists(new):
            print(f"{name}: MISSING in {a.new}")
            bad += 1
            continue
        d = diff(strip(load(ref)), strip(load(new)))
        if d:
            bad += 1
            print(f"{name}: {len(d)} differences")
            for line in d[:10]:
                print("   ", line)
        else:
            print(f"{name}: identical")
    print(f"{len(pairs) - bad}/{len(pairs)} result files identical to the reference (timings and environment ignored)")
    if bad:
        sys.exit(2)


if __name__ == "__main__":
    main()
