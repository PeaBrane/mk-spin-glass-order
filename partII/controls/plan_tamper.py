#!/usr/bin/env python3
"""Control harness for the plan-label check: run check.py with a tampered n-fold plan.

  python3 partII/controls/plan_tamper.py "1,1,2;2,1,4;4,4,8" --n 8 --point partII/inputs/point_n8.json \
      --threads 1 --kernel python --out out/x.json

The untrusted constructor construct.nfold_plan is replaced by the given plan (steps "i1,i2,k" separated by ";");
check.py itself is unchanged.  A plan whose labels are wrong must be rejected by the trusted exact.check_nfold_plan
(exit status 2, "CHECK FAILED: plan: ...") before any bridge level is computed.
"""
import os
import sys

PART = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PART)

import check  # noqa: E402
from mkcert import construct, exact  # noqa: E402

plan = [tuple(int(v) for v in step.split(",")) for step in sys.argv[1].split(";")]
construct.nfold_plan = lambda n: list(plan)
sys.argv = [os.path.join(PART, "check.py")] + sys.argv[2:]
try:
    check.main()
except exact.CertificateError as ex:
    print("CHECK FAILED:", ex, flush=True)
    sys.exit(2)
