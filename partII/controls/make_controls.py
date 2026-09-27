#!/usr/bin/env python3
"""Must-fail controls for Part II: deliberately invalid inputs that the checkers must reject (exit status 2).

  python3 partII/controls/make_controls.py OUTDIR      # writes the inputs and OUTDIR/controls_partII.json

Bridge controls (check.py, pure-Python kernel):
  nc1_entry_early        point_n8 with J lowered by one (nu_{J-1} does not enter the family)   -> entry check fails
  nc2_beta_minus_0.1     point_n8 with beta = 1/2 - 1/10 = 2/5 (T = 5/2), same schedule         -> entry check fails
  nc4_family_hash        point_n8 whose recorded family SHA-256 is wrong                         -> hash check fails
  nc5_wrong_n_cli        --n 7 on point_n8                                                       -> n binding fails
  nc6_wrong_n_file       point_n8 relabelled n = 7 (family file is for n = 8)                    -> n binding fails
  nc7_json_float         point_n8 with "J": 19.0                                                 -> input rejected
  nc12_plan_k_ne_sum     n-fold plan (1,1,2),(2,1,4),(4,4,8): a label with k != i1 + i2          -> plan check fails
  nc13_plan_short        n-fold plan ending at 4 summands for n = 8                             -> plan check fails
Family controls (check_family_stdlib.py):
  nc3_family_lam         family_n8 with lam = 2 instead of 1493/1000                             -> (C2) fails
  nc8_json_float_mass    family_n4 with one X_c mass written as a JSON float                     -> input rejected
  nc9_family_C1          family_n4 with G_4 atom 0 -> 1 unit and its last mass - 1 (valid law)   -> (C1) fails at t = 0
  nc10_law_sha_mismatch  the same change without updating the law's recorded sha256              -> input rejected
  nc11_family_eps0       family_n8 with eps0 = 1/30 (x10)                                        -> (C2) fails
Manifest control (tools/manifest.py):
  nc14_manifest_hash     input manifest with one wrong SHA-256                                   -> verification fails
"""
import copy
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PART = os.path.dirname(HERE)
REPO = os.path.dirname(PART)
INPUTS = os.path.join(PART, "inputs")
sys.path.insert(0, PART)

from mkcert.exact import law_sha256  # noqa: E402


def _load(name):
    with open(os.path.join(INPUTS, name)) as fh:
        return json.load(fh)


def _write(outdir, name, obj):
    path = os.path.join(outdir, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")
    return path


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def make(outdir, py=sys.executable):
    os.makedirs(outdir, exist_ok=True)
    check = os.path.join(PART, "check.py")
    famck = os.path.join(PART, "check_family_stdlib.py")
    ctl = []

    def point_ctl(name, obj, n, expect):
        path = _write(outdir, name + ".json", obj)
        ctl.append({"name": name, "slow": expect == "entry fails",
                    "argv": [py, check, "--n", str(n), "--point", path, "--threads", "1", "--kernel", "python",
                             "--out", os.path.join(outdir, name + ".out.json")], "expect": expect})

    def fam_ctl(name, obj, n, expect):
        path = _write(outdir, name + ".json", obj)
        ctl.append({"name": name, "argv": [py, famck, "--n", str(n), "--family", path, "--out",
                                           os.path.join(outdir, name + ".out.json")], "expect": expect})

    # the point controls reference the family by basename, so a copy of family_n8.json sits next to them
    fam8 = _load("family_n8.json")
    _write(outdir, "family_n8.json", fam8)
    pt = _load("point_n8.json")

    p = copy.deepcopy(pt)
    p["J"] -= 1
    p["coarsen"] = p["coarsen"][: p["J"]]
    point_ctl("nc1_entry_early", p, 8, "entry fails")
    p = copy.deepcopy(pt)
    p["beta"], p["T"] = "2/5", "5/2"
    point_ctl("nc2_beta_minus_0.1", p, 8, "entry fails")
    p = copy.deepcopy(pt)
    p["family"]["sha256"] = hashlib.sha256(b"not the family file").hexdigest()
    point_ctl("nc4_family_hash", p, 8, "family file SHA-256 does not match")
    ctl.append({"name": "nc5_wrong_n_cli",
                "argv": [py, check, "--n", "7", "--point", os.path.join(INPUTS, "point_n8.json"), "--threads", "1",
                         "--kernel", "python", "--out", os.path.join(outdir, "nc5.out.json")],
                "expect": "point file is for n = 8, requested n = 7"})
    p = copy.deepcopy(pt)
    p["n"] = 7
    point_ctl("nc6_wrong_n_file", p, 7, "family: file is for n = 8, requested n = 7")
    p = copy.deepcopy(pt)
    p["J"] = float(p["J"])
    point_ctl("nc7_json_float", p, 8, "J: must be a JSON integer")
    for name, plan, expect in (("nc12_plan_k_ne_sum", "1,1,2;2,1,4;4,4,8", "has k != i1 + i2"),
                               ("nc13_plan_short", "1,1,2;2,2,4", "last step produces 4 summands")):
        ctl.append({"name": name,
                    "argv": [py, os.path.join(HERE, "plan_tamper.py"), plan, "--n", "8", "--point",
                             os.path.join(INPUTS, "point_n8.json"), "--threads", "1", "--kernel", "python", "--out",
                             os.path.join(outdir, name + ".out.json")], "expect": expect})

    f = copy.deepcopy(fam8)
    f["lam"] = "2"
    fam_ctl("nc3_family_lam", f, 8, "(C2) fails")
    f = copy.deepcopy(fam8)
    f["eps0"] = "1/30"
    fam_ctl("nc11_family_eps0", f, 8, "(C2) fails")
    fam4 = _load("family_n4.json")
    f = copy.deepcopy(fam4)
    f["Xc"]["m"][0] = float(f["Xc"]["m"][0])
    fam_ctl("nc8_json_float_mass", f, 4, "X_c: every mass must be a JSON integer")
    f = copy.deepcopy(fam4)
    g = f["G"]["4"]
    g["atom"] += 1
    g["m"][-1] -= 1
    fam_ctl("nc10_law_sha_mismatch", copy.deepcopy(f), 4, "recorded law sha256 does not match")
    g["sha256"] = law_sha256(g["atom"], g["m"])
    fam_ctl("nc9_family_C1", f, 4, "(C1) fails at t = 0.0")

    lines = []
    with open(os.path.join(INPUTS, "SHA256SUMS.txt")) as fh:
        for i, line in enumerate(fh):
            digest, rel = line.split(None, 1)
            path = os.path.abspath(os.path.join(INPUTS, rel.strip()))
            if i == 0:
                digest = hashlib.sha256(b"not this file").hexdigest()
            lines.append(f"{digest}  {path}\n")
    man = os.path.join(outdir, "SHA256SUMS_tampered.txt")
    with open(man, "w") as fh:
        fh.writelines(lines)
    ctl.append({"name": "nc14_manifest_hash", "argv": [py, os.path.join(REPO, "tools", "manifest.py"), "verify", man],
                "expect": "FAILED"})
    with open(os.path.join(outdir, "controls_partII.json"), "w") as fh:
        json.dump(ctl, fh, indent=1)
    return ctl


if __name__ == "__main__":
    out = sys.argv[1]
    print(f"{len(make(out))} Part II controls written to {out}")
