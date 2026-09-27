"""Level-0 check: the staircase law nu_0 dominates the law of beta |J|, J ~ N(0,1)  (trusted code).

Claim checked: CDF_{nu_0}(t) >= erf(t / (beta sqrt 2)) for every t >= 0.
On bin k, phi(t) = erf(t/(beta sqrt 2)) - l_k(t) with l_k the linear CDF of nu_0 is concave, so for any tangent
point t_c the line phi(t_c) + phi'(t_c)(t - t_c) majorises phi; the check is that this line is <= 0 at both bin
ends, evaluated with interval enclosures (mkcert.ivdec) and decided on the upper endpoints.  The tangent point is
a float guess rounded to a dyadic rational (any point of the bin is valid).  Bins where CDF_{nu_0} = 1 need no
check (erf < 1), and at t = 0 both sides are >= 0.  Since P(beta'|J| <= t) <= P(beta|J| <= t) for beta' >= beta,
the same nu_0 is valid for every beta' >= beta.
"""
import math
from fractions import Fraction

from . import ivdec as V
from .exact import ONE, cdf_knots, need


def check_gauss(atom, m, beta, e):
    beta = Fraction(beta)
    N = len(m)
    C = cdf_knots(atom, m)
    h = Fraction(2) ** (-e)
    bs2 = V.IV.of(beta) * V.sqrt(V.IV.of(2))
    dens0 = V.sqrt(V.IV.of(2) / V.pi()) / beta  # density of beta|J| at 0
    d0f = float(dens0.hi)
    hf = float(h)
    worst = None
    for k in range(N):
        if C[k] == ONE:
            break
        slope = Fraction(m[k], ONE) / h
        L, R = h * k, h * (k + 1)
        sf = float(slope)
        if sf <= 0:
            tcf = (k + 1) * hf
        elif d0f <= sf:
            tcf = k * hf
        else:
            tcf = min(max(float(beta) * math.sqrt(2 * math.log(d0f / sf)), k * hf), (k + 1) * hf)
        tc = min(max(Fraction(tcf), L), R)  # exact dyadic tangent point in the bin
        ell = Fraction(C[k], ONE) + slope * (tc - L)
        phi = V.erf(V.IV.of(tc) / bs2) - V.IV.of(ell)
        dphi = dens0 * V.exp(-V.IV.of(tc * tc / (2 * beta * beta))) - V.IV.of(slope)
        up_r = (phi + dphi * V.IV.of(R - tc)).hi_frac()
        up_l = (phi + dphi * V.IV.of(L - tc)).hi_frac()
        need(up_r <= 0 and up_l <= 0, f"gauss: bin {k}")
        v = max(up_r, up_l)
        worst = v if worst is None else max(worst, v)
    return worst
