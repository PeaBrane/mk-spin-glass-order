#!/usr/bin/env python3
"""Copy selected SEARCH outputs into inputs/ under canonical names, pointing at the family file by basename and
dropping the search diagnostics `search_info` (optional; used when regenerating inputs with search.py).

  python freeze_points.py <search_dir> <inputs_dir> <tag>=<search point basename> ...
"""
import json
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
for arg in sys.argv[3:]:
    tag, name = arg.split("=")
    with open(os.path.join(src, name)) as fh:
        d = json.load(fh)
    d["family"]["file"] = os.path.basename(d["family"]["file"])
    d.pop("search_info", None)
    out = os.path.join(dst, f"point_{tag}.json")
    with open(out, "w") as fh:
        json.dump(d, fh, indent=1)
        fh.write("\n")
    print(out, d["n"], d["beta"], d["J"], d["grid"], d["branch"])
