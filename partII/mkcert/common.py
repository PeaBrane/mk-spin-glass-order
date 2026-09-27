"""File formats, hashing and environment helpers (standard library only)."""
import hashlib
import json
import os
import platform
import sys
from fractions import Fraction

from .exact import CertificateError, law_sha256, need

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hashes():
    """SHA-256 of every .py file of the verifier (package and scripts)."""
    out = {}
    for root in (CODE_DIR, os.path.join(CODE_DIR, "mkcert")):
        for name in sorted(os.listdir(root)):
            if name.endswith(".py"):
                p = os.path.join(root, name)
                out[os.path.relpath(p, CODE_DIR)] = sha256_file(p)
    return out


def _reject_constant(name):
    raise CertificateError(f"JSON constant {name} is not allowed")


def load_json(path):
    """Parse a JSON file; NaN and +-Infinity are rejected.  JSON numbers with a fraction or exponent become Python
    floats, which every certificate field rejects (see as_int, frac, law_from_json)."""
    with open(path) as fh:
        return json.load(fh, parse_constant=_reject_constant)


def dump_json(obj, path):
    tmp = path + ".partial"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=_default)
        fh.write("\n")
    os.replace(tmp, path)


def _default(x):
    if isinstance(x, Fraction):
        return str(x)
    raise TypeError(type(x))


def as_int(v, name):
    """A certificate integer: must be a JSON integer (Python int, not bool); nothing is coerced."""
    need(type(v) is int, f"{name}: must be a JSON integer, got {type(v).__name__}")
    return v


def law_from_json(d, name):
    """Staircase law from JSON: integer atom and masses (no coercion); the optional per-law 'sha256' is verified."""
    need(isinstance(d, dict) and set(d) >= {"atom", "m"}, f"{name}: malformed law")
    need(isinstance(d["m"], list), f"{name}: masses must be a JSON list")
    atom = as_int(d["atom"], f"{name}.atom")
    need(all(type(v) is int for v in d["m"]), f"{name}: every mass must be a JSON integer")
    m = list(d["m"])
    if "sha256" in d:
        need(d["sha256"] == law_sha256(atom, m), f"{name}: recorded law sha256 does not match its content")
    return atom, m


def law_to_json(atom, m):
    return {"atom": int(atom), "m": [int(v) for v in m], "sha256": law_sha256(atom, m)}


def frac(s, name):
    """Parse an exact rational written as 'p/q' or an integer string (or a JSON integer); floats are rejected."""
    need(type(s) in (str, int), f"{name}: must be an exact rational string")
    if isinstance(s, str):
        need("." not in s and "e" not in s.lower(), f"{name}: decimal/float notation not allowed")
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        raise CertificateError(f"{name}: not a rational number: {s!r}")


def environment(mods=("numpy", "numba", "flint")):
    env = {"python": sys.version.split()[0], "system": platform.system()}
    for mod in mods:
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    return env


def peak_rss_mb():
    """Peak resident set size of this process in MB: max of getrusage and /proc VmHWM (whichever is available)."""
    vals = []
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        vals.append(r / 1024.0 if sys.platform != "darwin" else r / (1024.0 * 1024.0))
    except Exception:
        pass
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    vals.append(int(line.split()[1]) / 1024.0)
    except OSError:
        pass
    return max(vals) if vals else None


__all__ = ["CertificateError", "sha256_file", "code_hashes", "load_json", "dump_json", "as_int", "law_from_json",
           "law_to_json", "frac", "environment", "peak_rss_mb"]
