"""Finite-scale bridge (Theorem 1B): iterate exact staircase lower laws until the family is entered.

nu_0 = Gaussian law of beta|J| on h_0 = 2^-e0 (N bins).  Level step: branch law (f = atanh(tanh a tanh b))
then n-fold conv (binary doubling Lemma-M chain).  Before a step the grid may be coarsened h -> 2h.
Entry test at level j (exact Fractions): CDF_{nu_j}(t) <= alpha0 + (1-alpha0) F_c(t/L0) for all t,
with L0 = c0_up/eps0 and alpha0 = A c0_lo / L0, c0 = (ln 2)/2.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from fractions import Fraction

import numba as nb
import numpy as np
from flint import arb

import rc_core as rc
from rc_core import ONE, Law

C0_UP = Fraction(34657360, 10 ** 8)
C0_LO = Fraction(34657359, 10 ** 8)


def check_c0_bounds():
    c0 = arb(2).log() / 2
    lo = rc.arb_frac(C0_LO)
    up = rc.arb_frac(C0_UP)
    assert (c0 - lo) > 0 and (up - c0) > 0, "c0 rational bounds invalid"


def entry_check(nu, e, Xc_m, Nx, hc, L0, alpha0, exact=True):
    """Return (ok, worst) ; worst = max over tested knots of CDF_nu - target (float, diagnostic)."""
    h = Fraction(1, 2 ** e)
    Cn = [int(v) for v in nu.cdf()]
    mn = [int(v) for v in nu.m]
    N = nu.N
    Cc = [0]
    for v in Xc_m[:Nx]:
        Cc.append(Cc[-1] + int(v))
    mc = [int(v) for v in Xc_m[:Nx]]
    tmax = L0 * Nx * hc
    # float pre-screen (diagnostic only)
    kmax = min(N, int(float(tmax / h)) + 1)
    tk = np.arange(kmax + 1) * float(h)
    Cn_f = np.array(Cn[: kmax + 1], dtype=float) / ONE
    Cc_f = np.array(Cc, dtype=float) / ONE
    tau = tk / float(L0 * hc)
    Fc_f = np.interp(tau, np.arange(Nx + 1), Cc_f, right=1.0)
    targ_f = float(alpha0) + (1 - float(alpha0)) * Fc_f
    worst_f = float(np.max(Cn_f - targ_f))
    if worst_f > 1e-9 or not exact:
        return False, worst_f

    def cdf_nu(t):
        q = t / h
        k = q.numerator // q.denominator
        if k >= N:
            return Fraction(1)
        return Fraction(Cn[k], ONE) + Fraction(mn[k], ONE) * (q - k)

    def Fc(tau):
        k = tau.numerator // tau.denominator
        if k >= Nx:
            return Fraction(1)
        return Fraction(Cc[k], ONE) + Fraction(mc[k], ONE) * (tau - k)

    worst = None
    k = 0
    while True:
        t = k * h
        if t > tmax:
            break
        val = Fraction(Cn[k], ONE) if k <= N else Fraction(1)
        targ = alpha0 + (1 - alpha0) * Fc(t / (L0 * hc))
        d = val - targ
        if worst is None or d > worst:
            worst = d
        if d > 0:
            return False, float(d)
        k += 1
        if k > N:
            break
    for kc in range(Nx + 1):
        t = L0 * kc * hc
        targ = alpha0 + (1 - alpha0) * Fraction(Cc[kc], ONE)
        d = cdf_nu(t) - targ
        if d > worst:
            worst = d
        if d > 0:
            return False, float(d)
    return True, float(worst)


def sha256_file(p):
    hh = hashlib.sha256()
    with open(p, "rb") as fh:
        hh.update(fh.read())
    return hh.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--beta", required=True)
    ap.add_argument("--e0", type=int, default=8)
    ap.add_argument("--N", type=int, default=8000)
    ap.add_argument("--s", type=int, default=4)
    ap.add_argument("--sb", type=int, default=4)
    ap.add_argument("--R", type=int, default=14)
    ap.add_argument("--dmax", type=float, default=12.0)
    ap.add_argument("--family", required=True, help="prefix of rc_family output (.npz/.json)")
    ap.add_argument("--maxlev", type=int, default=400)
    ap.add_argument("--coarsen_med_frac", type=float, default=0.1)
    ap.add_argument("--coarsen_tail_frac", type=float, default=0.75)
    ap.add_argument("--coarsen_tail_thr", type=float, default=2.0 ** -30)
    ap.add_argument("--fail_median", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_every", type=int, default=1)
    a = ap.parse_args()
    nb.set_num_threads(a.threads)
    check_c0_bounds()
    os.makedirs(a.out, exist_ok=True)
    fam = json.load(open(a.family + ".json"))
    famz = np.load(a.family + ".npz")
    assert fam["C1"] and fam["C2"], "family certificate not verified"
    assert fam["n"] == a.n
    Xc_m = famz["Xc_m"]
    Nx = int(fam["Nx"])
    hc = Fraction(fam["xmax"]) / Nx
    A = Fraction(fam["A"])
    eps0 = Fraction(fam["eps0"])
    L0 = C0_UP / eps0
    alpha0 = A * C0_LO / L0
    beta = Fraction(a.beta)
    meta = dict(vars(a))
    meta.update({"L0": str(L0), "alpha0": str(alpha0), "P": rc.P, "arb_prec": rc.ctx.prec,
                 "family_json_sha256": sha256_file(a.family + ".json"),
                 "family_npz_sha256": sha256_file(a.family + ".npz"),
                 "rc_core_sha256": sha256_file(rc.__file__),
                 "rc_bridge_sha256": sha256_file(__file__),
                 "python": sys.version, "numpy": np.__version__, "numba": nb.__version__})
    import flint
    meta["python_flint"] = flint.__version__
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "levels.log"), "a")
    t0 = time.time()
    e = a.e0
    nu = rc.gauss_law(beta, e, a.N)
    status = "running"
    J = None
    for j in range(a.maxlev + 1):
        tl = time.time()
        ok, worst = entry_check(nu, e, Xc_m, Nx, hc, L0, alpha0)
        med = nu.median_bin() * 2.0 ** -e
        rec = {"level": j, "e": e, "median": med, "atom": nu.atom / ONE, "entry_worst": worst,
               "entry_ok": ok}
        if j % a.save_every == 0 or ok:
            np.savez_compressed(os.path.join(a.out, f"nu_{j:04d}.npz"), e=np.int64(e),
                                atom=np.int64(nu.atom), m=nu.m)
        if ok:
            status = "CERTIFIED"
            J = j
            rec["time_total_s"] = time.time() - t0
            print(json.dumps(rec), file=log, flush=True)
            print(json.dumps(rec), flush=True)
            break
        if med < a.fail_median:
            status = "FAILED_flows_to_0"
            print(json.dumps(rec), file=log, flush=True)
            break
        if j == a.maxlev:
            status = "FAILED_maxlev"
            print(json.dumps(rec), file=log, flush=True)
            break
        # coarsening decision
        C = nu.cdf()
        ncoarse = 0
        while True:
            mb = nu.median_bin()
            tail = ONE - int(C[int(a.N * a.coarsen_tail_frac)])
            if mb > a.coarsen_med_frac * a.N or tail > a.coarsen_tail_thr * ONE:
                nu = rc.coarsen(nu)
                e -= 1
                ncoarse += 1
                C = nu.cdf()
                np.savez_compressed(os.path.join(a.out, f"nu_{j:04d}_c{ncoarse}.npz"), e=np.int64(e),
                                    atom=np.int64(nu.atom), m=nu.m)
            else:
                break
        rec["coarsened"] = ncoarse
        diag = {}
        tb = time.time()
        Y = rc.branch_law(nu, e, s=a.s, sb=a.sb, R=a.R, dmax=a.dmax, diag=diag)
        rec["t_branch"] = time.time() - tb
        tc = time.time()
        out, ops, pw = rc.nfold(Y, a.n)
        rec["t_conv"] = time.time() - tc
        rec.update(diag)
        np.savez_compressed(os.path.join(a.out, f"step_{j:04d}.npz"), e=np.int64(e),
                            Y_atom=np.int64(Y.atom), Y_m=Y.m,
                            **{f"pw{k}_atom": np.int64(v.atom) for k, v in pw.items()},
                            **{f"pw{k}_m": v.m for k, v in pw.items()},
                            out_atom=np.int64(out.atom), out_m=out.m,
                            ops=np.array(json.dumps(ops)))
        nu = out
        rec["t_level"] = time.time() - tl
        print(json.dumps(rec), file=log, flush=True)
        print(json.dumps(rec), flush=True)
    summ = {"status": status, "J": J, "levels_run": j, "e_final": e, "time_total_s": time.time() - t0}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
