"""The finite-scale bridge (Theorem 1B): one engine for SEARCH and CHECK.

Level step j -> j+1 (grid h = 2^-e, N bins):
  1. optional coarsening h -> 2h (count frozen per level in the certificate);
  2. branch: U = rigorous fine bounds of f(A,B), A, B iid ~ nu_j (mkcert.branch); Y = staircase built from U;
  3. n-fold sum: binary-doubling chain of `conv` (construct.nfold_plan); nu_{j+1} = the n-branch sum.
In CHECK mode the labels of the n-fold plan are verified first (exact.check_nfold_plan: k = i1 + i2 at every step,
no label reused, the last step produces n), nu_{j+1} is the output of the plan's last
step, and every constructed law is validated and every inequality is re-verified:
  level 0 against the Gaussian (gauss.check_gauss), coarsening (exact.check_coarsen), branch output against U
  (exact.check_branch_output), every conv (exact.check_conv), and the family entry at level J
  (exact.check_entry).  The phi_l bounds for the Theorem 2/3 constants are taken from nu_0..nu_J.
"""
import hashlib
import time
from fractions import Fraction

from . import branch, constants, construct, exact, gauss
from .exact import ONE, need


def grid_h(e):
    return Fraction(2) ** (-e)


def entry_prescreen(nu, e, Xc_m, Nx, hc, L0, alpha0):
    """Float pre-screen of the entry inequality (SEARCH only; the decision is re-made exactly)."""
    h = 2.0 ** (-e)
    lh = float(L0 * hc)
    tmax = float(L0 * Nx * hc)
    C = nu.cdf()
    Cc = [0]
    for v in Xc_m[:Nx]:
        Cc.append(Cc[-1] + v)
    worst = -1.0
    for k in range(min(nu.N, int(tmax / h) + 1) + 1):
        tau = k * h / lh
        j = int(tau)
        Fc = 1.0 if j >= Nx else (Cc[j] + Xc_m[j] * (tau - j)) / ONE
        worst = max(worst, C[k] / ONE - (float(alpha0) + (1 - float(alpha0)) * Fc))
    return worst


def run(n, beta, e0, N, s, sb, R, dmax, fam, L0, alpha0, *, mode, kernel, schedule=None, J=None,
        maxlev=2000, log=print, coarsen_rule=None, fail_median=1e-3):
    """mode = 'search' (decide coarsening, stop at the first entry) or 'check' (follow the frozen schedule and
    verify every inequality).  Returns a result dict."""
    need(mode in ("search", "check"), "mode")
    verify = mode == "check"
    if verify:
        need(schedule is not None and J is not None and len(schedule) == J, "check needs the frozen schedule")
    Xc_m = fam["Xc"][1]
    Nx, hc = fam["Nx"], fam["hc"]
    plan = [tuple(step) for step in construct.nfold_plan(n)]
    if verify:
        exact.check_nfold_plan(plan, n)
    t0 = time.time()
    e = e0
    nu = construct.gauss_law(beta, e, N)
    counts = {"gauss_bins": 0, "coarsen_ops": 0, "branch_ops": 0, "branch_fine_points": 0, "conv_ops": 0,
              "conv_bins": 0, "entry_checks": 0}
    rec0 = {"level": 0}
    if verify:
        exact.validate(*nu.pair(), "nu_0")
        rec0["gauss_worst_upper"] = float(gauss.check_gauss(*nu.pair(), beta, e))
        counts["gauss_bins"] = N
    chain = hashlib.sha256()
    phis = []
    levels = []
    sched = []
    status = "running"
    worst = {"branch": None, "conv": None}
    j = 0
    while True:
        tl = time.time()
        pair = nu.pair()
        chain.update(f"{j}:{e}:".encode() + exact.law_bytes(*pair))
        phi, _, _ = constants.phi_bridge(*pair, e)
        phis.append(phi)
        rec = rec0 if j == 0 else {"level": j}
        rec.update({"e": e, "median_bin": nu.median_bin(), "atom": pair[0] / ONE, "phi_upper": float(phi)})
        # ---- entry into the family
        if mode == "search":
            pre = entry_prescreen(nu, e, Xc_m, Nx, hc, L0, alpha0)
            rec["entry_prescreen"] = pre
            if pre <= 1e-9:
                try:
                    rec["entry_margin"] = float(exact.check_entry(pair, grid_h(e), fam["Xc"], hc, Nx, L0, alpha0))
                    status, J = "CERTIFIED", j
                except exact.CertificateError:
                    pass
            if status == "CERTIFIED":
                levels.append(rec)
                log(rec)
                break
            if nu.median_bin() * 2.0 ** (-e) < fail_median:
                status = "FAILED_flows_to_0"
                levels.append(rec)
                log(rec)
                break
            if j >= maxlev:
                status = "FAILED_maxlev"
                levels.append(rec)
                log(rec)
                break
        elif j == J:
            rec["entry_margin"] = float(exact.check_entry(pair, grid_h(e), fam["Xc"], hc, Nx, L0, alpha0))
            counts["entry_checks"] += 1
            status = "CERTIFIED"
            levels.append(rec)
            log(rec)
            break
        # ---- coarsening
        if mode == "search":
            c = 0
            while coarsen_rule(nu):
                nu = construct.coarsen(nu)
                e -= 1
                c += 1
        else:
            c = int(schedule[j])
            for _ in range(c):
                new = construct.coarsen(nu)
                exact.validate(*new.pair(), f"coarsened law at level {j}")
                exact.check_coarsen(nu.pair(), new.pair())
                counts["coarsen_ops"] += 1
                nu = new
                e -= 1
        sched.append(c)
        rec["coarsen"] = c
        # ---- branch
        tb = time.time()
        U, binfo = branch.fine_bounds(nu.atom, nu.m, e, s, sb, R, dmax, kernel=kernel)
        rec["t_branch_bounds"] = time.time() - tb
        Y = construct.branch_from_bounds(U, N, s)
        if verify:
            exact.validate(*Y.pair(), f"Y at level {j}")
            sl = exact.check_branch_output(Y.pair(), U, s)
            worst["branch"] = sl if worst["branch"] is None else min(worst["branch"], sl)
            counts["branch_ops"] += 1
            counts["branch_fine_points"] += N * s
        rec["Omax"] = binfo["Omax"]
        # ---- n-fold sum
        tc = time.time()
        laws = {1: Y}
        for (i1, i2, k) in plan:
            Z = construct.conv(laws[i1], laws[i2])
            if verify:
                exact.validate(*Z.pair(), f"conv {k} at level {j}")
                sl = exact.check_conv(laws[i1].pair(), laws[i2].pair(), Z.pair())
                worst["conv"] = sl if worst["conv"] is None else min(worst["conv"], sl)
                counts["conv_ops"] += 1
                counts["conv_bins"] += N
            laws[k] = Z
        rec["t_conv"] = time.time() - tc
        need(plan[-1][2] == n and n in laws, "n-fold sum: last plan step does not produce n summands")
        nu = laws[plan[-1][2]]
        rec["t_level"] = time.time() - tl
        levels.append(rec)
        log(rec)
        j += 1
    return {
        "status": status,
        "J": J if status == "CERTIFIED" else None,
        "levels_run": j,
        "schedule": sched,
        "e_final": e,
        "phis": phis,
        "chain_sha256": chain.hexdigest(),
        "levels": levels,
        "worst_branch_slack": worst["branch"],
        "worst_conv_slack": worst["conv"],
        "counts": counts,
        "time_s": time.time() - t0,
    }


def default_coarsen_rule(N, med_frac=0.1, tail_frac=0.75, tail_thr=2.0 ** -30):
    """SEARCH heuristic: coarsen while the median passes 0.1 N or the tail beyond 0.75 N > 2^-30."""
    def rule(nu):
        tail = ONE - nu.cdf()[int(N * tail_frac)]
        return nu.median_bin() > med_frac * N or tail > tail_thr * ONE
    return rule
