"""Rigorous transcendental constants via Arb (python-flint) balls.

Only the exact endpoints of a ball are used: a ball is [mid - rad, mid + rad] with mid and rad exact dyadic
numbers, which are converted to Fractions without rounding.  Every decision in the certificate is taken on
these endpoints (upper endpoints for quantities bounded from above, lower endpoints for those bounded from
below).  Floating point is never used in a claimed inequality.
"""
from fractions import Fraction

from flint import arb, ctx

from mkcert.exact import need

ctx.prec = 192

# Rational bounds for c0 = (ln 2)/2 and for ln2/2 + 1/4 = int_0^infty psi (verified below with Arb).
C0_LO = Fraction(34657359, 10 ** 8)
C0_UP = Fraction(34657360, 10 ** 8)
LN2H_Q_UP = Fraction(59657360, 10 ** 8)


def dyadic(x):
    """Exact Fraction of an exact arb (mid or rad of a ball)."""
    m, e = x.man_exp()
    m, e = int(m), int(e)
    return Fraction(m << e) if e >= 0 else Fraction(m, 1 << (-e))


def ball(x):
    """(lo, hi) exact Fraction endpoints of the Arb ball x."""
    if not x.is_finite():
        raise ArithmeticError("non-finite Arb ball")
    mid = dyadic(x.mid())
    rad = dyadic(x.rad())
    return mid - rad, mid + rad


def hi(x):
    return ball(x)[1]


def lo(x):
    return ball(x)[0]


def ceil_scaled(x, S):
    """Smallest integer >= every point of the ball x times 2^S."""
    v = hi(x) * (Fraction(2) ** S)
    return -((-v.numerator) // v.denominator)


def floor_scaled(x, S):
    """Largest integer <= every point of the ball x times 2^S."""
    v = lo(x) * (Fraction(2) ** S)
    return v.numerator // v.denominator


def arb_frac(fr):
    fr = Fraction(fr)
    return arb(fr.numerator) / arb(fr.denominator)


def verify_constants():
    """Check the rational bounds C0_LO < (ln 2)/2 < C0_UP and ln2/2 + 1/4 < LN2H_Q_UP on Arb endpoints."""
    c0 = arb(2).log() / 2
    lo0, hi0 = ball(c0)
    need(C0_LO < lo0 and hi0 < C0_UP, "c0 bounds")
    need(hi(c0 + arb(1) / 4) < LN2H_Q_UP, "ln2/2 + 1/4 bound")
    return {"C0_LO": str(C0_LO), "C0_UP": str(C0_UP), "LN2H_Q_UP": str(LN2H_Q_UP), "arb_prec": ctx.prec}
