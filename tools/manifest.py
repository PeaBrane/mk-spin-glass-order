#!/usr/bin/env python3
"""SHA-256 manifests in sha256sum format (standard library only).

  python3 tools/manifest.py verify partII/inputs/SHA256SUMS.txt   # exit 2 on any mismatch or missing file
  python3 tools/manifest.py make DIR [--glob '*.json'] > DIR/SHA256SUMS.txt   # paths relative to DIR, sorted
"""
import argparse
import fnmatch
import hashlib
import os
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_verify(a):
    base = os.path.dirname(os.path.abspath(a.manifest))
    bad = 0
    count = 0
    with open(a.manifest) as fh:
        for line in fh:
            if not line.strip():
                continue
            digest, rel = line.rstrip("\n").split(None, 1)
            rel = rel.lstrip("*").strip()
            path = os.path.join(base, rel)
            count += 1
            if not os.path.exists(path):
                print(f"{rel}: MISSING")
                bad += 1
            elif sha256_file(path) != digest:
                print(f"{rel}: FAILED")
                bad += 1
            else:
                print(f"{rel}: OK")
    print(f"{a.manifest}: {count - bad}/{count} OK")
    if bad or count == 0:
        sys.exit(2)


def cmd_make(a):
    out = []
    for root, _, files in os.walk(a.dir):
        for f in files:
            if fnmatch.fnmatch(f, a.glob):
                p = os.path.join(root, f)
                out.append((os.path.relpath(p, a.dir).replace(os.sep, "/"), sha256_file(p)))
    for rel, digest in sorted(out):
        print(f"{digest}  {rel}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("manifest")
    m = sub.add_parser("make")
    m.add_argument("dir")
    m.add_argument("--glob", default="*.json")
    a = ap.parse_args()
    {"verify": cmd_verify, "make": cmd_make}[a.cmd](a)


if __name__ == "__main__":
    main()
