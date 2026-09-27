"""Rigorous interval enclosures with the Python standard library (trusted code).

An interval is a pair (lo, hi) of `decimal.Decimal` numbers with lo <= x <= hi for the real number x it encloses.
Only the four basic operations are used, each evaluated twice: once in a context that rounds toward -infinity
(for lower endpoints) and once toward +infinity (upper endpoints).  The decimal module implements these
roundings exactly as specified (General Decimal Arithmetic, IEEE 754-2008 directed rounding).  No library
transcendental function is used: exp, log, atan and erf are summed from their Taylor series with explicit
remainder bounds, and square roots are verified by squaring.  Results are converted to Fractions exactly.

Enclosures (x a rational or an interval):
  exp   : halve the argument m times so |y| <= 1/8, sum the Taylor series until the term t_k is below
          10^-(prec+3), bound the remainder by |t_k| (the tail ratio is <= |y|/(k+2) <= 1/16), then square m
          times (positive intervals, directed rounding);
  log   : for x > 0, x = x' 2^j with x' in [1/2, 1), log x' = 2 atanh((x'-1)/(x'+1)), |z| <= 1/3 (+ rounding);
          atanh summed until the term t_k is below 10^-(prec+3), remainder <= |t_k| a^2/(1-a^2) <= 0.131 |t_k|
          for a = |z| <= 0.34; log 2 = 2 atanh(1/3);
  pi    : Machin, 16 atan(1/5) - 4 atan(1/239), alternating series with remainder <= first omitted term;
  erf   : for 0 <= x <= 7, (2/sqrt(pi)) sum_k (-1)^k x^{2k+1}/(k!(2k+1)), summed until k > x^2 + 2 (tested
          exactly on the rational value of x) and the term is < 10^-(prec+5); from then on the terms decrease
          (ratio < x^2/(k+2) < 1), so the alternating tail is bounded by the next term;
          for x > 7, 1 - e^{-x^2}/(x sqrt(pi)) <= erf(x) <= 1 (valid for every x > 0);
  sqrt  : decimal square root, adjusted by next_plus / next_minus until the squares bracket the argument.

Where floats appear (none decides an inequality):
  _exp_point  chooses the number m of halvings from float(x); the reduced argument is then tested exactly
              (|y| <= 1/8, else ArithmeticError), so a float error can only raise, never give a wrong enclosure;
  _log_point  guesses the binary exponent j from float(x); j is then corrected by exact Decimal comparisons until
              x 2^-j lies in [1/2, 1);
  _erf_point  chooses the branch (series for x <= 7, tail bound for x > 7) and the working precision from
              float(x); both branches are valid for every x > 0 and precision only affects the width.  The
              series stopping rule k > x^2 + 2, which the alternating-tail bound needs, is an exact rational test
              (an earlier revision used the float x^2; the two agree on every input used here).
"""
import math
from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal
from fractions import Fraction

PREC = 64


_CTX = {}


def _ctx(prec):
    """(round-down context, round-up context) with `prec` significant digits."""
    if prec not in _CTX:
        _CTX[prec] = (Context(prec=prec, rounding=ROUND_FLOOR, Emin=-10 ** 9, Emax=10 ** 9),
                      Context(prec=prec, rounding=ROUND_CEILING, Emin=-10 ** 9, Emax=10 ** 9))
    return _CTX[prec]


class IV:
    """Closed interval [lo, hi] of Decimals, with arithmetic in `prec` significant digits."""

    __slots__ = ("lo", "hi", "prec")

    def __init__(self, lo, hi, prec=PREC):
        if not lo <= hi:
            raise ArithmeticError("empty interval")
        self.lo, self.hi, self.prec = lo, hi, prec

    # ------------------------------------------------------------------ constructors
    @staticmethod
    def of(q, prec=PREC):
        """Enclosure of the rational q (int, Fraction or 'p/q' string)."""
        q = Fraction(q)
        dn, up = _ctx(prec)
        a, b = Decimal(q.numerator), Decimal(q.denominator)
        return IV(dn.divide(a, b), up.divide(a, b), prec)

    def _coerce(self, other):
        return other if isinstance(other, IV) else IV.of(other, self.prec)

    # ------------------------------------------------------------------ arithmetic
    def __add__(self, o):
        o = self._coerce(o)
        dn, up = _ctx(self.prec)
        return IV(dn.add(self.lo, o.lo), up.add(self.hi, o.hi), self.prec)

    __radd__ = __add__

    def __sub__(self, o):
        o = self._coerce(o)
        dn, up = _ctx(self.prec)
        return IV(dn.subtract(self.lo, o.hi), up.subtract(self.hi, o.lo), self.prec)

    def __rsub__(self, o):
        return self._coerce(o) - self

    def __neg__(self):
        return IV(-self.hi, -self.lo, self.prec)

    def __mul__(self, o):
        o = self._coerce(o)
        dn, up = _ctx(self.prec)
        ends = [(x, y) for x in (self.lo, self.hi) for y in (o.lo, o.hi)]
        return IV(min(dn.multiply(x, y) for x, y in ends), max(up.multiply(x, y) for x, y in ends), self.prec)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._coerce(o)
        if not o.lo > 0:
            raise ArithmeticError("division by an interval that is not positive")
        dn, up = _ctx(self.prec)
        ends = [(x, y) for x in (self.lo, self.hi) for y in (o.lo, o.hi)]
        return IV(min(dn.divide(x, y) for x, y in ends), max(up.divide(x, y) for x, y in ends), self.prec)

    def __rtruediv__(self, o):
        return self._coerce(o) / self

    def widen(self, r):
        """[lo - r, hi + r] for a nonnegative Decimal r."""
        dn, up = _ctx(self.prec)
        return IV(dn.subtract(self.lo, r), up.add(self.hi, r), self.prec)

    # ------------------------------------------------------------------ exact endpoints
    def lo_frac(self):
        return Fraction(self.lo)

    def hi_frac(self):
        return Fraction(self.hi)


def ceil_scaled(x, S):
    """Smallest integer >= every point of the interval x times 2^S."""
    v = x.hi_frac() * Fraction(2) ** S
    return -((-v.numerator) // v.denominator)


def floor_scaled(x, S):
    """Largest integer <= every point of the interval x times 2^S."""
    v = x.lo_frac() * Fraction(2) ** S
    return v.numerator // v.denominator


# ---------------------------------------------------------------------------------------- exp
def _mag(x):
    """Upper bound of |t| over the interval x."""
    return max(abs(x.lo), abs(x.hi))


def _exp_point(x, prec):
    """Enclosure of exp(x) for a Decimal x: reduce to |y| <= 1/8, Taylor series, square back."""
    xf = abs(float(x))
    m = 0 if xf <= 0.125 else int(math.ceil(math.log2(xf / 0.125))) + 1
    dn, up = _ctx(prec)
    p2 = Decimal(2 ** m)
    y = IV(dn.divide(x, p2), up.divide(x, p2), prec)
    if not _mag(y) <= Decimal("0.125"):
        raise ArithmeticError("exp: argument reduction failed")
    s = IV(Decimal(1), Decimal(1), prec)
    term = IV(Decimal(1), Decimal(1), prec)
    tiny = Decimal(10) ** (-(prec + 3))
    k = 0
    while True:
        k += 1
        term = term * y / k
        s = s + term
        if _mag(term) < tiny:
            break
    # remainder: sum_{i>=1} |term_k| (|y|/(k+2))^i <= |term_k| since |y|/(k+2) <= 1/16
    s = s.widen(_mag(term))
    if not s.lo > 0:
        raise ArithmeticError("exp: nonpositive lower bound")
    for _ in range(m):
        s = IV(dn.multiply(s.lo, s.lo), up.multiply(s.hi, s.hi), prec)
    return s


def exp(x):
    """Enclosure of exp over the interval (or rational) x; exp is increasing."""
    if not isinstance(x, IV):
        x = IV.of(x)
    return IV(_exp_point(x.lo, x.prec).lo, _exp_point(x.hi, x.prec).hi, x.prec)


# ---------------------------------------------------------------------------------------- atanh, log
def _atanh_series(z):
    """Enclosure of atanh(z) for an interval z with |z| <= 0.34 (odd series + remainder)."""
    prec = z.prec
    a = _mag(z)
    if not a <= Decimal("0.34"):
        raise ArithmeticError("atanh: argument too large")
    z2 = z * z
    s = z
    pw = z
    tiny = Decimal(10) ** (-(prec + 3))
    k = 0
    term = z
    while _mag(term) >= tiny:
        k += 1
        pw = pw * z2
        term = pw / (2 * k + 1)
        s = s + term
    # remainder: sum_{i>=1} |term_k| a^{2i} = |term_k| a^2/(1-a^2) <= 0.131 |term_k|  (a <= 0.34)
    _, up = _ctx(prec)
    return s.widen(up.multiply(_mag(term), Decimal("0.131")))


_LN2 = {}


def ln2(prec=PREC):
    if prec not in _LN2:
        _LN2[prec] = _atanh_series(IV.of(Fraction(1, 3), prec)) * 2
    return _LN2[prec]


def _log_point(x, prec):
    """Enclosure of log(x) for a Decimal x > 0."""
    if not x > 0:
        raise ArithmeticError("log of nonpositive number")
    j = math.frexp(float(x))[1] if float(x) > 0 else int(math.floor(float(x.logb()) * math.log2(10))) + 1
    # x' = x 2^{-j} should lie in [1/2, 1); adjust j exactly with Decimal comparisons
    dn, up = _ctx(prec)

    def scaled(jj):
        p = Decimal(2 ** abs(jj))  # exact power of two
        if jj >= 0:
            return IV(dn.divide(x, p), up.divide(x, p), prec)
        return IV(dn.multiply(x, p), up.multiply(x, p), prec)

    xs = scaled(j)
    while xs.hi >= 1:
        j += 1
        xs = scaled(j)
    while xs.lo < Decimal("0.5"):
        j -= 1
        xs = scaled(j)
    z = (xs - 1) / (xs + 1)
    return _atanh_series(z) * 2 + ln2(prec) * j


def log(x):
    """Enclosure of log over the positive interval (or rational) x; log is increasing."""
    if not isinstance(x, IV):
        x = IV.of(x)
    return IV(_log_point(x.lo, x.prec).lo, _log_point(x.hi, x.prec).hi, x.prec)


# ---------------------------------------------------------------------------------------- pi, sqrt
def _atan_inv(q, prec):
    """Enclosure of atan(1/q) for an integer q >= 5 (alternating series, remainder <= next term)."""
    z = IV.of(Fraction(1, q), prec)
    z2 = z * z
    s = z
    pw = z
    k = 0
    while True:
        k += 1
        pw = pw * z2
        t = pw / (2 * k + 1)
        s = s + t if k % 2 == 0 else s - t
        if t.hi < Decimal(10) ** (-(prec + 5)):
            nxt = (pw * z2 / (2 * k + 3)).hi
            return s.widen(nxt)


_PI = {}


def pi(prec=PREC):
    if prec not in _PI:
        _PI[prec] = _atan_inv(5, prec) * 16 - _atan_inv(239, prec) * 4
    return _PI[prec]


def sqrt(x):
    """Enclosure of sqrt over the nonnegative interval x (verified by squaring)."""
    if not isinstance(x, IV):
        x = IV.of(x)
    dn, up = _ctx(x.prec)
    if x.lo < 0:
        raise ArithmeticError("sqrt of negative")
    hi = up.sqrt(x.hi)
    while dn.multiply(hi, hi) < x.hi:
        hi = up.next_plus(hi)
    lo = dn.sqrt(x.lo)
    while up.multiply(lo, lo) > x.lo:
        lo = dn.next_minus(lo)
    return IV(lo, hi, x.prec)


# ---------------------------------------------------------------------------------------- erf
def _erf_point(x, prec):
    """Enclosure of erf(x) for a Decimal x >= 0."""
    if x < 0:
        raise ArithmeticError("erf: negative argument")
    xf = float(x)
    if xf > 7.0:
        xi = IV(x, x, prec)
        tail = exp(-(xi * xi)) / (xi * sqrt(pi(prec)))
        dn, _ = _ctx(prec)
        return IV(dn.subtract(Decimal(1), tail.hi), Decimal(1), prec)
    p2 = prec + int(math.ceil(xf * xf * 0.4343)) + 10  # digits lost to cancellation <= x^2 log10(e)
    xsq = Fraction(x) * Fraction(x)  # exact square of the Decimal x, for the stopping rule
    xi = IV(x, x, p2)
    x2 = xi * xi
    p = xi  # x^{2k+1} / k!
    s = xi
    k = 0
    tiny = Decimal(10) ** (-(prec + 5))
    while True:
        k += 1
        p = p * x2 / k
        t = p / (2 * k + 1)
        s = s + t if k % 2 == 0 else s - t
        if t.hi < tiny and k > xsq + 2:  # exact test; terms decrease from index k+1 on (k + 2 > x^2)
            nxt = (p * x2 / (k + 1) / (2 * k + 3)).hi  # first omitted term (terms decrease: k + 1 > x^2)
            s = s.widen(nxt)
            break
    two_over_sqrtpi = IV.of(2, p2) / sqrt(pi(p2))
    r = s * two_over_sqrtpi
    dn, up = _ctx(prec)
    return IV(dn.plus(r.lo), up.plus(r.hi), prec)


def erf(x):
    """Enclosure of erf over the nonnegative interval (or rational) x; erf is increasing."""
    if not isinstance(x, IV):
        x = IV.of(x)
    return IV(_erf_point(x.lo, x.prec).lo, _erf_point(x.hi, x.prec).hi, x.prec)
