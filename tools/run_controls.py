#!/usr/bin/env python3
"""Run every must-fail control of both parts and record the outcomes (standard library only).

  python3 tools/run_controls.py --out out/controls --jobs 4 [--skip-slow]

A control succeeds when the checker exits with status 2 and prints the expected "CHECK FAILED" reason (for the
manifest control: status 2 and a FAILED line).  Writes OUT/controls_summary.json; exit status 2 if any control
was accepted or failed for an unexpected reason.  --skip-slow omits the two Part II controls that run a full
bridge before failing (about 60 s each).
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(ctl):
    t0 = time.time()
    p = subprocess.run(ctl["argv"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    lines = p.stdout.strip().splitlines()
    msg = next((ln for ln in reversed(lines) if "CHECK FAILED" in ln or "FAILED" in ln), lines[-1] if lines else "")
    msg = msg.replace(REPO + os.sep, "")
    ok = p.returncode == 2 and (ctl["expect"] is None or ctl["expect"] in msg)
    return {"name": ctl["name"], "exit": p.returncode, "message": msg[:300], "expect": ctl["expect"],
            "as_expected": ok, "wall_s": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--skip-slow", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    m1 = _module(os.path.join(REPO, "partI", "controls", "make_controls.py"), "controls_partI")
    m2 = _module(os.path.join(REPO, "partII", "controls", "make_controls.py"), "controls_partII")
    ctls = [dict(c, part="I") for c in m1.make(os.path.join(a.out, "partI"))]
    ctls += [dict(c, part="II") for c in m2.make(os.path.join(a.out, "partII"))
             if not (a.skip_slow and c.get("slow"))]
    ctls.sort(key=lambda c: not c.get("slow"))  # start the slow ones first
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        res = list(ex.map(run, ctls))
    for c, r in zip(ctls, res):
        r["part"] = c["part"]
    res.sort(key=lambda r: (r["part"], r["name"]))
    bad = [r for r in res if not r["as_expected"]]
    for r in res:
        flag = "rejected as expected" if r["as_expected"] else "UNEXPECTED"
        print(f"Part {r['part']} {r['name']}: exit {r['exit']}, {flag}: {r['message']}")
    summary = {"kind": "mk-controls-summary", "count": len(res), "unexpected": len(bad),
               "python": sys.version.split()[0], "wall_s": round(time.time() - t0, 1), "controls": res}
    with open(os.path.join(a.out, "controls_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
        fh.write("\n")
    print(f"{len(res)} controls, {len(bad)} unexpected outcomes ({summary['wall_s']} s)")
    if bad:
        sys.exit(2)


if __name__ == "__main__":
    main()
