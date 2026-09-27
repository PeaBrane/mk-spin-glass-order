#!/usr/bin/env python3
"""CHECK Part I certificates (Prop A + Lemma T + Transfer-Theorem constants); standard library only.

  # every frozen input listed in a manifest (hashes verified first), one result file per input:
  python3 partI/check_partI.py --manifest partI/inputs/SHA256SUMS.txt --out-dir out/partI
  # one input; --n must equal the n recorded in the file:
  python3 partI/check_partI.py --input partI/inputs/oneshot/oneshot_n4.json --n 4 --out out/n4.json

Exit status 0: every certificate and every claim verified.  Exit status 2 and a line "CHECK FAILED: ..." on the
first failure (hash mismatch, malformed input, failed hypothesis of Prop A / Lemma T, or a false claim).
The mathematics and the arithmetic model are described in propa.py and in the repository README.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import propa  # noqa: E402
from propa import CertificateError, need  # noqa: E402

CODE_FILES = ["partI/check_partI.py", "partI/propa.py", "partII/mkcert/ivdec.py", "partII/mkcert/exact.py",
              "partII/mkcert/__init__.py"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reject_constant(name):
    raise CertificateError(f"JSON constant {name} is not allowed")


def load(path):
    with open(path) as fh:
        return json.load(fh, parse_constant=_reject_constant)


def read_manifest(path):
    """Lines '<sha256>  <relative path>' (sha256sum format); paths are relative to the manifest's directory."""
    entries = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            digest, rel = line.rstrip("\n").split(None, 1)
            entries.append((digest, rel.lstrip("*").strip()))
    need(entries, "manifest is empty")
    return entries


def environment():
    return {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "system": platform.system()}


def check_one(path, rel, n_expected=None):
    t0 = time.time()
    cert = load(path)
    if n_expected is not None:
        need(cert.get("n") == n_expected, f"input is for n = {cert.get('n')}, requested n = {n_expected}")
    res = propa.check(cert)
    return {
        "kind": "mk-partI-check", "verdict": "PASS",
        "input": {"file": rel, "sha256": sha256_file(path)},
        "result": res,
        "code": {f: sha256_file(os.path.join(REPO, f)) for f in CODE_FILES},
        "environment": environment(),
        "wall_s": time.time() - t0,
    }


def dump(obj, path):
    tmp = path + ".partial"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="sha256sum-style manifest of the inputs to check")
    ap.add_argument("--out-dir", help="output directory (with --manifest)")
    ap.add_argument("--input", help="a single input file")
    ap.add_argument("--n", type=int, help="n of --input (must match the file)")
    ap.add_argument("--out", help="output file (with --input)")
    a = ap.parse_args()
    t0 = time.time()
    if a.manifest:
        need(a.out_dir and not a.input, "use --manifest with --out-dir")
        base = os.path.dirname(os.path.abspath(a.manifest))
        entries = read_manifest(a.manifest)
        for digest, rel in entries:
            need(os.path.isfile(os.path.join(base, rel)), f"missing input {rel}")
            got = sha256_file(os.path.join(base, rel))
            need(got == digest, f"sha256 mismatch for {rel}: manifest {digest}, file {got}")
        os.makedirs(a.out_dir, exist_ok=True)
        summary = {"kind": "mk-partI-summary", "manifest": {"file": os.path.basename(a.manifest),
                                                            "sha256": sha256_file(a.manifest)},
                   "results": {}}
        for digest, rel in entries:
            rec = check_one(os.path.join(base, rel), rel)
            name = rel.replace("/", "__").replace(".json", "") + ".result.json"
            dump(rec, os.path.join(a.out_dir, name))
            r = rec["result"]
            summary["results"][rel] = {"verdict": "PASS", "n": r["n"], "beta": r["beta"], "claims": r["claims"],
                                       "K": r["K"], "pole": r["pole_lower6"], "EA_fixed": r["EA_fixed_lower6"],
                                       "EQ2": r["EQ2_lower6"], "ED2": r["ED2_lower6"], "result_file": name}
            print(f"PASS {rel}: n={r['n']} beta={r['beta']} K={r['K']} claims={','.join(r['claims'])} "
                  f"pole>={r['pole_lower6']} EA>={r['EA_fixed_lower6']} EQ2>={r['EQ2_lower6']} "
                  f"ED2>={r['ED2_lower6']} ({rec['wall_s']:.2f}s)", flush=True)
        summary["verdict"] = "ALL PART I CHECKS PASSED"
        summary["count"] = len(entries)
        summary["environment"] = environment()
        summary["wall_s"] = time.time() - t0
        dump(summary, os.path.join(a.out_dir, "partI_summary.json"))
        print(f"{summary['verdict']} ({len(entries)} certificates, {summary['wall_s']:.1f}s)", flush=True)
        return
    need(a.input and a.out and a.n is not None, "use --input with --n and --out")
    rec = check_one(os.path.abspath(a.input), os.path.basename(a.input), n_expected=a.n)
    dump(rec, a.out)
    r = rec["result"]
    print(f"PASS {a.input}: n={r['n']} beta={r['beta']} K={r['K']} claims={','.join(r['claims'])} "
          f"EQ2>={r['EQ2_lower6']} ED2>={r['ED2_lower6']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except CertificateError as ex:
        print("CHECK FAILED:", ex, flush=True)
        sys.exit(2)
