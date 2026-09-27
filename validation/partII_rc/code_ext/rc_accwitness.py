"""Sidecar: intermediate n-fold accumulator witnesses for a frozen rc_bridge run.

rc_core.nfold combines the binary powers Y^{*k} into an accumulator; for n with two or more
non-doubling additions (n = 7: 4+2 -> 6, 6+1 -> 7) rc_bridge saves only the final law, so the frozen
rc_check cannot verify the chain.  This script replays each saved `ops` list with rc_core.conv on the
saved powers pw_k, asserts the replay reproduces the saved `out` law bit for bit, and writes every
intermediate accumulator to <acc_dir>/acc_JJJJ.npz (keys acc{k}_atom, acc{k}_m).  The witnesses are
not trusted: rc_check_acc.py re-verifies each one with its own exact conv check.

--corrupt_level j writes, for level j only, the final n-fold law in place of every intermediate
accumulator (a deliberately invalid witness, used as a negative control).
"""
import argparse
import glob
import json
import os

import numpy as np

import rc_core as rc
from rc_core import Law


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--acc_dir", required=True)
    ap.add_argument("--corrupt_level", type=int, default=None)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.run, "meta.json")))
    n = int(meta["n"])
    os.makedirs(a.acc_dir, exist_ok=True)
    steps = sorted(glob.glob(os.path.join(a.run, "step_*.npz")))
    levels = [a.corrupt_level] if a.corrupt_level is not None else [int(p[-8:-4]) for p in steps]
    nsaved = 0
    for j in levels:
        st = np.load(os.path.join(a.run, f"step_{j:04d}.npz"))
        ops = json.loads(str(st["ops"]))
        pw = {1: Law(int(st["Y_atom"]), st["Y_m"])}
        for k in [2, 4, 8, 16]:
            if f"pw{k}_m" in st.files:
                pw[k] = Law(int(st[f"pw{k}_atom"]), st[f"pw{k}_m"])
        out = Law(int(st["out_atom"]), st["out_m"])
        acc, accn = None, 0
        inter = {}
        for _, k1, k2, k3 in ops:
            if k1 == k2 and acc is None:
                continue  # doubling op, its output pw[k3] is saved by rc_bridge
            src = acc if acc is not None else pw[k1]
            assert accn == k1 or acc is None, f"level {j}: op chain mismatch"
            acc = rc.conv(src, pw[k2])
            accn = k3
            if k3 != n:
                inter[k3] = acc
        if acc is None:
            acc = pw[n]
        assert accn in (0, n) and acc.atom == out.atom and bool((acc.m == out.m).all()), \
            f"level {j}: replay does not reproduce saved out"
        if a.corrupt_level is not None:
            inter = {k: out for k in inter}
        if not inter:
            continue
        np.savez_compressed(os.path.join(a.acc_dir, f"acc_{j:04d}.npz"),
                            **{f"acc{k}_atom": np.int64(v.atom) for k, v in inter.items()},
                            **{f"acc{k}_m": v.m for k, v in inter.items()})
        nsaved += 1
    print(json.dumps({"run": a.run, "n": n, "levels": len(levels), "files_saved": nsaved,
                      "corrupt_level": a.corrupt_level}))


if __name__ == "__main__":
    main()
