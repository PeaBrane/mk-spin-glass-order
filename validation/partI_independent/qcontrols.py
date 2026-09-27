"""Must-fail controls for qcert (own verifier): each mutates one thing in a frozen input; all must be rejected."""
import copy
import glob
import json
import sys
from fractions import Fraction

import qcert


def verdict(cert):
    try:
        return qcert.check(cert)["all_claims_hold"]
    except qcert.CertFail:
        return False


def main(root):
    rows = []
    for f in sorted(glob.glob(f"{root}/thresholds/*.json") + glob.glob(f"{root}/oneshot/*.json")
                    + glob.glob(f"{root}/bounds/*.json")):
        base = json.load(open(f))
        name = "/".join(f.split("/")[-2:])
        muts = {}
        c = copy.deepcopy(base)
        c["beta"] = str(Fraction(base["beta"]) - Fraction(1, 10))
        muts["beta-0.1"] = c
        c = copy.deepcopy(base)
        c["tail"]["gamma"] = str(Fraction(base["tail"]["gamma"]) * (1 - Fraction(1, 10**9)))
        muts["gamma*(1-1e-9)"] = c
        c = copy.deepcopy(base)
        c["tail"]["A"] = str(Fraction(base["tail"]["A"]) / 1000)
        muts["A/1000"] = c
        c = copy.deepcopy(base)
        c["n"] = base["n"] - 1
        muts["n-1"] = c
        if base["levels"]:
            c = copy.deepcopy(base)
            c["levels"] = base["levels"][:-1]
            muts["drop_last_level"] = c
        for k, c in muts.items():
            v = verdict(c)
            rows.append({"input": name, "mutation": k, "accepted": v})
            print(json.dumps(rows[-1]), flush=True)
    acc = [r for r in rows if r["accepted"]]
    print(json.dumps({"controls": len(rows), "wrongly_accepted": len(acc)}))


if __name__ == "__main__":
    main(sys.argv[1])
