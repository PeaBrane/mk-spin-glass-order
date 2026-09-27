"""Independent checker: re-evaluates every certificate inequality from the saved raw laws.

Separate code paths from rc_core / rc_family / rc_bridge:
* level 0: CDF_nu0 >= erf(t/(beta sqrt2)) on every bin via a tangent-line bound of the concave gap
  (mpmath interval arithmetic, not Arb);
* coarsen: new CDF >= old CDF at all old knots (exact integers);
* conv: output CDF >= EXACT piecewise-quadratic CDF of |X+Y|, computed from the unsigned
  sum/difference decomposition (p*q and p*rev(q)), checked per bin including the vertex;
* min2: output CDF >= exact 1-(1-F)^2 per bin including the vertex;
* branch: U'_i >= P(f(A,B) <= u_i) recomputed with delta tables from mpmath.iv and a separately
  written numba kernel with 16-bit limbs; check CDF_Y(u_i) >= U'_{i+1} for all fine i;
  plus a random spot check of the identity x >= atanh(tanh u / tanh b) with mpmath.iv directly;
* family (C1)/(C2) and the entry dominance re-evaluated with exact rationals.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
from fractions import Fraction

import mpmath
import numba as nb
import numpy as np
from flint import arb, ctx, fmpz_poly
ctx.prec = 192

ONE = 1 << 48
iv = mpmath.iv
iv.prec = 200


class Fail(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Fail(msg)


def load_law(z, prefix=""):
    atom = int(z[prefix + "atom"])
    m = z[prefix + "m"].astype(np.int64)
    return atom, m


def valid(atom, m, name):
    need(atom >= 0, f"{name}: negative atom")
    need(bool((m >= 0).all()), f"{name}: negative mass")
    need(bool((m[1:] <= m[:-1]).all()), f"{name}: masses increase")
    need(atom + sum(int(v) for v in m) == ONE, f"{name}: total != 2^48")


def cdf_list(atom, m):
    C = [atom]
    for v in m:
        C.append(C[-1] + int(v))
    return C


# ----------------------------------------------------------------------------- level 0
def iv_ceil_scaled(x, S):
    b = x.b
    v = mpmath.mpf(b) * mpmath.mpf(2) ** S
    c = int(mpmath.ceil(v))
    return c


def check_gauss(atom, m, beta, e):
    """CDF_nu0(t) >= erf(t/(beta sqrt 2)) for all t in [0, N h]; tangent-line bound per bin (Arb)."""
    N = len(m)
    C = cdf_list(atom, m)
    b = arb(beta.numerator) / beta.denominator
    h = arb(1) / arb(2) ** e
    cden = (arb(2) / arb.pi()).sqrt() / b
    cf = float(cden.mid())
    worst = -1.0
    for k in range(N):
        if C[k] == ONE:
            break  # ell == 1 on this and all later bins, and erf(x) < 1 for every finite x
        slope = arb(int(m[k])) / ONE / h
        L = h * k
        R = h * (k + 1)
        sf = float(slope.mid())
        if sf <= 0:
            tc = R
        elif cf <= sf:
            tc = L
        else:
            ts = float(beta) * math.sqrt(2 * math.log(cf / sf))
            ts = min(max(ts, k * 2.0 ** -e), (k + 1) * 2.0 ** -e)
            tc = arb(ts)  # any point of the bin works for the tangent bound
        # phi(t) = erf(t/(b sqrt2)) - ell(t) is concave: phi(t) <= phi(tc) + phi'(tc)(t - tc)
        ell_tc = arb(C[k]) / ONE + arb(int(m[k])) / ONE * (tc - L) / h
        phi_tc = (tc / (b * arb(2).sqrt())).erf() - ell_tc
        dphi = cden * (-(tc * tc) / (2 * b * b)).exp() - slope
        up1 = phi_tc + dphi * (R - tc)
        up2 = phi_tc + dphi * (L - tc)
        mx = max(float(up1.upper()), float(up2.upper()))
        need(up1.upper() <= 0 and up2.upper() <= 0, f"gauss: bin {k} erf exceeds CDF by <= {mx}")
        worst = max(worst, mx)
    return worst


# ----------------------------------------------------------------------------- coarsen
def check_coarsen(old, new):
    (a0, m0), (a1, m1) = old, new
    N = len(m0)
    C0 = cdf_list(a0, m0)
    C1 = cdf_list(a1, m1)
    for k in range(N + 1):
        if k % 2 == 0:
            v = C1[k // 2] if k // 2 <= N else ONE
            need(v >= C0[k], f"coarsen: knot {k}")
        else:
            j = k // 2
            need(C1[j] + C1[j + 1] >= 2 * C0[k], f"coarsen: mid knot {k}")
    return True


# ----------------------------------------------------------------------------- conv
def _coeffs(poly, L):
    cs = [int(v) for v in poly.coeffs()]
    return cs + [0] * (L - len(cs))


def check_conv(X, Y, Z):
    """CDF_Z >= exact CDF of |X+Y| (X, Y independent s.u. staircase laws)."""
    (a, p), (b, q), (z0, zm) = X, Y, Z
    N = len(p)
    p = [int(v) for v in p]
    q = [int(v) for v in q]
    S = _coeffs(fmpz_poly(p) * fmpz_poly(q), 2 * N)  # S[k-1] = D_sum at knot k
    T = _coeffs(fmpz_poly(p) * fmpz_poly(q[::-1]), 2 * N)  # T[k + N - 1] = D_diff at knot k

    def Dsum(k):
        return S[k - 1] if 1 <= k <= 2 * N - 1 else 0

    def Ddiff(k):
        i = k + N - 1
        return T[i] if 0 <= i < 2 * N - 1 else 0

    E = [Dsum(k) + Ddiff(k) + Ddiff(-k) for k in range(N + 1)]
    Zc = cdf_list(z0, zm)
    Q4 = 4 * a * b
    worst = None
    for k in range(N):
        lin = 4 * a * q[k] + 4 * b * p[k] + 2 * E[k]
        quad = E[k + 1] - E[k]
        d0 = 4 * ONE * Zc[k] - Q4
        d1 = 4 * ONE * int(zm[k]) - lin
        d2 = quad
        need(d0 >= 0, f"conv: knot {k}")
        need(d0 + d1 - d2 >= 0, f"conv: knot {k + 1}")
        if d2 < 0 and 2 * d2 < d1 < 0:
            need(d1 * d1 <= -4 * d0 * d2, f"conv: interior of bin {k}")
        Q4 += lin + quad
        dd = d0 if worst is None else min(worst, d0)
        worst = dd
    return worst / (4 * ONE * ONE)


def check_min2(X, Z):
    (a, m), (z0, zm) = X, Z
    N = len(m)
    C = cdf_list(a, m)
    Zc = cdf_list(z0, zm)
    for k in range(N):
        mk = int(m[k])
        d0 = ONE * Zc[k] - (ONE * ONE - (ONE - C[k]) ** 2)
        d1 = ONE * int(zm[k]) - 2 * (ONE - C[k]) * mk
        d2 = -mk * mk  # Delta = d0 + d1 s - d2 s^2 with -d2 = +mk^2 (convex)
        need(d0 >= 0, f"min2: knot {k}")
        need(d0 + d1 - d2 >= 0, f"min2: knot {k + 1}")
        if d2 < 0 and 2 * d2 < d1 < 0:
            need(d1 * d1 <= -4 * d0 * d2, f"min2: interior {k}")
    return True


# ----------------------------------------------------------------------------- branch
_TAB = {}


def tables_mp(e, s, R, mmax):
    key = (e, s, R, mmax)
    if key in _TAB:
        return _TAB[key]
    scale = mpmath.mpf(2) ** (R + e)
    hf = iv.mpf(1) / (iv.mpf(2) ** e * s)
    up = np.zeros(mmax + 1, dtype=np.int64)
    for mm in range(1, mmax + 1):
        d = hf * mm
        val = -iv.log(1 - iv.exp(-2 * d)) / 2
        up[mm] = int(mpmath.ceil(mpmath.mpf(val.b) * scale))
    lo = [0]
    mm = 1
    while True:
        d = hf * mm
        val = -iv.log(1 - iv.exp(-2 * d)) / 2
        v = int(mpmath.floor(mpmath.mpf(val.a) * scale))
        if v <= 0:
            break
        lo.append(v)
        mm += 1
    _TAB[key] = (up, np.array(lo, dtype=np.int64))
    return _TAB[key]


@nb.njit(parallel=True, cache=True)
def _chk_kernel(C, m, N, s, sb, R, Omax, Dup, Dlo, out):
    # out[i, 0..8]: 16-bit-limb partial sums of sum_jb m[jb//sb] * F_A(x_{i,jb})
    sig = s // sb
    msk = (1 << R) - 1
    stp = (1 << R) // s
    one = np.int64(1) << np.int64(48)
    L16 = np.int64(65535)
    nlo = Dlo.shape[0]
    M = out.shape[0]
    for i in nb.prange(M):
        jb0 = i // sig
        acc = np.zeros(9, dtype=np.int64)
        o = 1
        while o <= Omax:
            jb = jb0 + o
            if jb >= N * sb:
                break
            w = m[jb // sb]
            if w == 0:
                break
            x = np.int64(i) * stp + Dup[jb * sig - i]
            il = jb * sig + i
            if il < nlo:
                x = x - Dlo[il]
            bn = x >> R
            F = one
            if bn < N:
                F = C[bn] + ((m[bn] * (x & msk) + msk) >> R)
            w0 = w & L16
            w1 = (w >> 16) & L16
            w2 = w >> 32
            f0 = F & L16
            f1 = (F >> 16) & L16
            f2 = F >> 32
            acc[0] += w0 * f0
            acc[1] += w0 * f1
            acc[2] += w0 * f2
            acc[3] += w1 * f0
            acc[4] += w1 * f1
            acc[5] += w1 * f2
            acc[6] += w2 * f0
            acc[7] += w2 * f1
            acc[8] += w2 * f2
            o += 1
        for t in range(9):
            out[i, t] = acc[t]


def branch_U(A, e, s, sb, R, dmax):
    a, m = A
    N = len(m)
    C = np.array(cdf_list(a, m), dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    sig = s // sb
    Omax = min(int(dmax * sb * 2 ** e), N * sb, 1 << 14)
    need(Omax <= (1 << 14), "Omax")
    Dup, Dlo = tables_mp(e, s, R, (Omax + 1) * sig + 1)
    Clist = [int(v) for v in C]
    mlist = [int(v) for v in m]
    last = next(k for k in range(N + 1) if Clist[k] >= ONE)
    istop = min(N * s, last * s + s)
    out = np.zeros((istop, 9), dtype=np.int64)
    _chk_kernel(C, m, N, s, sb, R, Omax, Dup, Dlo, out)
    stp = (1 << R) // s
    Dt = int(Dup[Omax * sig + 1])
    U = [ONE] * (N * s + 2)
    for i in range(istop):
        jb0 = i // sig
        k0 = jb0 // sb
        if k0 >= N:
            continue
        head = sb * Clist[k0] + (jb0 % sb + 1) * mlist[k0]
        o = out[i]
        Ssum = (int(o[0]) + ((int(o[1]) + int(o[3])) << 16) + ((int(o[2]) + int(o[4]) + int(o[6])) << 32)
                + ((int(o[5]) + int(o[7])) << 48) + (int(o[8]) << 64))
        jbt = jb0 + Omax + 1
        tail = 0
        if jbt < N * sb:
            kt = jbt // sb
            TM = sb * ONE - (sb * Clist[kt] + (jbt % sb) * mlist[kt])
            if TM > 0:
                xt = i * stp + Dt
                bn = xt >> R
                Ft = ONE if bn >= N else Clist[bn] + ((mlist[bn] * (xt & ((1 << R) - 1)) + (1 << R) - 1) >> R)
                tail = TM * Ft
        num = head * ONE + Ssum + tail
        U[i] = min(-((-num) // (sb * ONE)), ONE)
    return U


def check_branch(A, Y, e, s, sb, R, dmax):
    U = branch_U(A, e, s, sb, R, dmax)
    ya, ym = Y
    N = len(ym)
    CY = cdf_list(ya, ym)
    worst = None
    for i in range(N * s):
        k, r = divmod(i, s)
        lhs = s * CY[k] + r * int(ym[k])
        rhs = s * U[i + 1]
        need(lhs >= rhs, f"branch: fine point {i}")
        d = lhs - rhs
        worst = d if worst is None or d < worst else worst
    return worst / (s * ONE), U


def spot_branch(A, e, s, sb, R, dmax, npts=40, seed=7):
    """Spot check: table-based x >= atanh(tanh u / tanh b) computed directly with Arb."""
    a, m = A
    N = len(m)
    sig = s // sb
    Omax = min(int(dmax * sb * 2 ** e), N * sb, 1 << 14)
    Dup, Dlo = tables_mp(e, s, R, (Omax + 1) * sig + 1)
    rng = random.Random(seed)
    for _ in range(npts):
        i = rng.randrange(0, N * s)
        o = rng.randrange(1, Omax + 1)
        jb = i // sig + o
        x = i * ((1 << R) // s) + int(Dup[jb * sig - i]) - (int(Dlo[jb * sig + i]) if jb * sig + i < len(Dlo) else 0)
        old_prec = ctx.prec
        ctx.prec = 192 + int(4 * (jb * sig) / (2 ** e * s))  # tanh near 1 needs ~2.9 bits per unit
        u = arb(i) / (arb(2) ** e * s)
        bb = arb(jb * sig) / (arb(2) ** e * s)
        g = (u.tanh() / bb.tanh()).atanh()
        xv = arb(x) / arb(2) ** (R + e)
        ok = bool(xv.lower() >= g.upper())
        ctx.prec = old_prec
        need(ok, f"spot: x < g at i={i} jb={jb}")
    return True


# ----------------------------------------------------------------------------- family and entry
def family_check(Xc_m, G, n, lam, A, eps0, hc, Nx):
    """(C1),(C2) re-evaluated on every elementary interval, both endpoints, interval slopes."""
    Cc = cdf_list(0, Xc_m)
    CG = {k: cdf_list(v[0], v[1]) for k, v in G.items()}
    mG = {k: [int(x) for x in v[1]] for k, v in G.items()}
    NG = len(G[1][1])

    def pl(C, mm, NN, tau):
        k = tau.numerator // tau.denominator
        if k >= NN:
            return Fraction(ONE)
        return C[k] + mm[k] * (tau - k)

    mc = [int(v) for v in Xc_m]
    tmax = lam * Nx
    pts = sorted(set([Fraction(k) for k in range(int(tmax) + 1)] + [lam * k for k in range(Nx + 1)]))
    pts = [t for t in pts if t <= tmax]
    epsI = eps0 / hc
    min_s, min_m = None, None
    for ta, tb in zip(pts[:-1], pts[1:]):
        kb = ta.numerator // ta.denominator
        sl = {k: Fraction(mG[k][kb]) if kb < NG else Fraction(0) for k in G}  # per index unit, x ONE
        for t in (ta, tb):
            Fc = pl(Cc, mc, Nx, t / lam) / ONE
            Gv = {k: pl(CG[k], mG[k], NG, t) / ONE for k in G}
            s = Fc - Gv[n]
            lhs = s + eps0 * A / lam * (1 - Fc)
            rhs = n * sl[n] / ONE / hc
            H = {k: Gv[k] + n * epsI * sl[k] / ONE for k in G}
            for kk in range(1, n + 1):
                lo = n - kk
                Ht = Fraction(1) if lo == 0 else max(H[j] for j in range(lo, n + 1))
                rhs += math.comb(n, kk) * (2 * A) ** kk * eps0 ** (kk - 1) * (Ht - Gv[n])
            marg = lhs - eps0 * rhs
            min_s = s if min_s is None else min(min_s, s)
            min_m = marg if min_m is None else min(min_m, marg)
    need(min_s >= 0, "C1 fails")
    need(min_m >= 0, "C2 fails")
    return min_s, min_m


def entry_recheck(nu, e, Xc_m, Nx, hc, L0, alpha0):
    a, m = nu
    N = len(m)
    C = cdf_list(a, m)
    mm = [int(v) for v in m]
    Cc = cdf_list(0, Xc_m)
    mc = [int(v) for v in Xc_m]
    h = Fraction(1, 2 ** e)
    tmax = L0 * Nx * hc

    def F_nu(t):
        q = t / h
        k = q.numerator // q.denominator
        return Fraction(ONE) if k >= N else C[k] + mm[k] * (q - k)

    def F_c(tau):
        k = tau.numerator // tau.denominator
        return Fraction(ONE) if k >= Nx else Cc[k] + mc[k] * (tau - k)

    pts = sorted(set([k * h for k in range(int(tmax / h) + 1)] + [L0 * hc * k for k in range(Nx + 1)]))
    worst = None
    for t in pts:
        if t > tmax:
            continue
        d = F_nu(t) / ONE - (alpha0 + (1 - alpha0) * F_c(t / (L0 * hc)) / ONE)
        worst = d if worst is None else max(worst, d)
        need(d <= 0, f"entry fails at t={float(t)}")
    return worst


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spot", type=int, default=20)
    a = ap.parse_args()
    nb.set_num_threads(a.threads)
    t0 = time.time()
    meta = json.load(open(os.path.join(a.run, "meta.json")))
    summ = json.load(open(os.path.join(a.run, "summary.json")))
    need(summ["status"] == "CERTIFIED", "run not certified")
    J = int(summ["J"])
    n = int(meta["n"])
    beta = Fraction(meta["beta"])
    s, sb, R, dmax = int(meta["s"]), int(meta["sb"]), int(meta["R"]), float(meta["dmax"])
    fam = json.load(open(meta["family"] + ".json"))
    famz = np.load(meta["family"] + ".npz")
    rep = {"run": a.run, "n": n, "beta": str(beta), "J": J, "checks": {}}
    # c0 bounds, independent (mpmath)
    c0 = iv.log(2) / 2
    C0_UP = Fraction(34657360, 10 ** 8)
    C0_LO = Fraction(34657359, 10 ** 8)
    need(c0.b < mpmath.mpf(C0_UP.numerator) / C0_UP.denominator, "c0_up")
    need(c0.a > mpmath.mpf(C0_LO.numerator) / C0_LO.denominator, "c0_lo")
    lam, A, eps0 = Fraction(fam["lam"]), Fraction(fam["A"]), Fraction(fam["eps0"])
    Nx = int(fam["Nx"])
    hc = Fraction(fam["xmax"]) / Nx
    L0 = C0_UP / eps0
    alpha0 = A * C0_LO / L0
    need(str(L0) == meta["L0"] and str(alpha0) == meta["alpha0"], "L0/alpha0 mismatch with run")
    # family
    Xc_m = famz["Xc_m"].astype(np.int64)
    valid(0, Xc_m, "Xc")
    need(bool((Xc_m[Nx:] == 0).all()), "Xc support beyond Nx")
    G = {}
    for k in range(1, n + 1):
        G[k] = (int(famz[f"G{k}_atom"]), famz[f"G{k}_m"].astype(np.int64))
        valid(*G[k], f"G{k}")
    check_min2((0, Xc_m), G[1])
    for k in range(2, n + 1):
        check_conv(G[k - 1], G[1], G[k])
    ms, mm = family_check(Xc_m[:Nx], G, n, lam, A, eps0, hc, Nx)
    rep["checks"]["family"] = {"C1_min_s": float(ms), "C2_min_margin_over_eps0": float(mm / eps0)}
    print("family ok", rep["checks"]["family"], flush=True)
    # level 0
    z = np.load(os.path.join(a.run, "nu_0000.npz"))
    nu = load_law(z)
    e = int(z["e"])
    need(e == int(meta["e0"]), "e0")
    valid(*nu, "nu0")
    gw = check_gauss(*nu, beta, e)
    rep["checks"]["gauss_worst"] = gw
    print("gauss ok", gw, flush=True)
    worst_branch = 1.0
    worst_conv = 1.0
    nsteps = 0
    for j in range(J):
        cf = sorted(glob.glob(os.path.join(a.run, f"nu_{j:04d}_c*.npz")), key=lambda p: int(p.split("_c")[-1][:-4]))
        for p in cf:
            zc = np.load(p)
            new = load_law(zc)
            valid(*new, p)
            need(int(zc["e"]) == e - 1, "coarsen e")
            check_coarsen(nu, new)
            nu, e = new, e - 1
        st = np.load(os.path.join(a.run, f"step_{j:04d}.npz"))
        need(int(st["e"]) == e, f"step {j} e")
        Y = load_law(st, "Y_")
        valid(*Y, f"Y{j}")
        wb, _ = check_branch(nu, Y, e, s, sb, R, dmax)
        if j % max(1, J // max(1, a.spot)) == 0:
            spot_branch(nu, e, s, sb, R, dmax, npts=10, seed=j)
        worst_branch = min(worst_branch, wb)
        ops = json.loads(str(st["ops"]))
        pw = {1: Y}
        for k in [2, 4, 8, 16]:
            if f"pw{k}_m" in st.files:
                pw[k] = load_law(st, f"pw{k}_")
                valid(*pw[k], f"pw{k}")
        acc, accn = None, 0
        for op in ops:
            _, k1, k2, k3 = op
            if k1 == k2 and k1 in pw and k3 in pw and (acc is None or accn != k1):
                worst_conv = min(worst_conv, check_conv(pw[k1], pw[k2], pw[k3]))
            else:
                src = acc if acc is not None and accn == k1 else pw[k1]
                tgt = load_law(st, "out_") if k3 == n else None
                need(tgt is not None, "intermediate accumulator not saved")
                worst_conv = min(worst_conv, check_conv(src, pw[k2], tgt))
                acc, accn = tgt, k3
        out = load_law(st, "out_")
        valid(*out, f"out{j}")
        final = pw[n] if n in pw else out
        need(final[0] == out[0] and bool((final[1] == out[1]).all()), "out != final pw")
        zn = np.load(os.path.join(a.run, f"nu_{j + 1:04d}.npz"))
        nxt = load_law(zn)
        need(int(zn["e"]) == e and nxt[0] == out[0] and bool((nxt[1] == out[1]).all()), f"chain break at {j}")
        nu = nxt
        nsteps += 1
        if j % 10 == 0:
            print(f"level {j} ok  e={e} worst_branch={worst_branch:.3e} t={time.time() - t0:.0f}s", flush=True)
    ew = entry_recheck(nu, e, Xc_m[:Nx], Nx, hc, L0, alpha0)
    rep["checks"]["entry_worst"] = float(ew)
    rep["checks"]["branch_min_slack"] = worst_branch
    rep["checks"]["conv_min_knot_slack"] = worst_conv
    rep["steps_checked"] = nsteps
    rep["verdict"] = "ALL CHECKS PASSED"
    rep["time_s"] = time.time() - t0
    rep["versions"] = {"python": sys.version, "mpmath": mpmath.__version__, "numpy": np.__version__,
                       "numba": nb.__version__, "mpmath_iv_prec": iv.prec}
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    try:
        main()
    except Fail as ex:
        print("CHECK FAILED:", ex)
        sys.exit(2)
