#!/usr/bin/env python3
"""Must-fail controls for Part I: deliberately invalid inputs that check_partI.py must reject (exit status 2).

  python3 partI/controls/make_controls.py OUTDIR      # writes the inputs and OUTDIR/controls_partI.json

Each control changes one thing in a frozen input:
  gamma_minus_1e-5_oneshot_n4    one-shot n = 4 certificate with gamma lowered by 10^-5         -> (T1) fails
  gamma_times_0.999_n4_order     n = 4 order/imbalance threshold certificate, gamma * 0.999      -> (T1) fails
  A_div_1000_oneshot_n4          one-shot n = 4 with A / 1000                                    -> (T0) or (T2) fails
  beta_minus_0.1_<file>          every threshold certificate at beta - 0.1, same parameters     -> a hypothesis or the claim fails
  drop_last_level_n4_order       n = 4 order certificate without its last level                -> tail fails
  wrong_n_file_oneshot_n4_to_3   one-shot n = 4 certificate relabelled n = 3 (kappa_3 = 11/10)   -> (T1) fails
  wrong_n_file_n5_order_to_4     n = 5 order certificate relabelled n = 4                        -> fails
  wrong_n_cli_oneshot_n4         --n 5 on the n = 4 file                                         -> n binding fails
  claim_order_on_pole_n4         the n = 4 one-shot pole certificate claiming 'order'           -> claim fails
  json_float_beta_oneshot_n4     beta written as the JSON number 525.0                           -> input rejected
  tampered_manifest_hash         manifest with one wrong SHA-256                                 -> hash check fails
"""
import copy
import hashlib
import json
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
PART = os.path.dirname(HERE)
INPUTS = os.path.join(PART, "inputs")
CHECK = os.path.join(PART, "check_partI.py")


def _load(rel):
    with open(os.path.join(INPUTS, rel)) as fh:
        return json.load(fh)


def _write(outdir, name, obj):
    path = os.path.join(outdir, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")
    return path


def _frac_str(q):
    q = Fraction(q)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def make(outdir, py=sys.executable):
    """Write the control inputs into outdir; return a list of {name, argv, expect} (expect: substring of the
    CHECK FAILED message, or None for any failure)."""
    os.makedirs(outdir, exist_ok=True)
    ctl = []

    def single(name, obj, n, expect=None):
        path = _write(outdir, name + ".json", obj)
        ctl.append({"name": name, "argv": [py, CHECK, "--input", path, "--n", str(n), "--out",
                                           os.path.join(outdir, name + ".out.json")], "expect": expect})

    c = _load("oneshot/oneshot_n4.json")
    t = copy.deepcopy(c)
    t["tail"]["gamma"] = _frac_str(Fraction(t["tail"]["gamma"]) - Fraction(1, 10 ** 5))
    single("gamma_minus_1e-5_oneshot_n4", t, 4, "(T1) fails")
    t = copy.deepcopy(c)
    t["tail"]["A"] = _frac_str(Fraction(t["tail"]["A"]) / 1000)
    single("A_div_1000_oneshot_n4", t, 4, None)
    t = copy.deepcopy(c)
    t["n"] = 3
    single("wrong_n_file_oneshot_n4_to_3", t, 3, "(T1) fails")
    t = copy.deepcopy(c)
    t["claims"] = ["order"]
    single("claim_order_on_pole_n4", t, 4, "claim 'order' fails")
    t = copy.deepcopy(c)
    t["beta"] = 525.0
    single("json_float_beta_oneshot_n4", t, 4, "beta: must be a string")
    ctl.append({"name": "wrong_n_cli_oneshot_n4",
                "argv": [py, CHECK, "--input", os.path.join(INPUTS, "oneshot", "oneshot_n4.json"), "--n", "5",
                         "--out", os.path.join(outdir, "wrong_n_cli.out.json")],
                "expect": "input is for n = 4, requested n = 5"})

    c = _load("thresholds/n4_order_imbalance.json")
    t = copy.deepcopy(c)
    t["tail"]["gamma"] = _frac_str(Fraction(t["tail"]["gamma"]) * Fraction(999, 1000))
    single("gamma_times_0.999_n4_order", t, 4, "(T1) fails")
    t = copy.deepcopy(c)
    t["levels"] = t["levels"][:-1]
    single("drop_last_level_n4_order", t, 4, None)

    t = _load("thresholds/n5_order.json")
    t["n"] = 4
    single("wrong_n_file_n5_order_to_4", t, 4, None)

    for rel in sorted(os.listdir(os.path.join(INPUTS, "thresholds"))):
        t = _load("thresholds/" + rel)
        t["beta"] = _frac_str(Fraction(t["beta"]) - Fraction(1, 10))
        single("beta_minus_0.1_" + rel[:-5], t, t["n"], None)

    # manifest whose first entry has a wrong digest (absolute paths to the real inputs)
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
    ctl.append({"name": "tampered_manifest_hash",
                "argv": [py, CHECK, "--manifest", man, "--out-dir", os.path.join(outdir, "tampered_manifest_out")],
                "expect": "sha256 mismatch"})
    with open(os.path.join(outdir, "controls_partI.json"), "w") as fh:
        json.dump(ctl, fh, indent=1)
    return ctl


if __name__ == "__main__":
    out = sys.argv[1]
    print(f"{len(make(out))} Part I controls written to {out}")
