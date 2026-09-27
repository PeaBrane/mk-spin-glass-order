#!/bin/sh
# Full verification of every certified number in the paper, with the Python standard library only.
#
#   sh run_all.sh [OUTDIR]            default OUTDIR = out
#   JOBS=5 sh run_all.sh              run up to 5 independent checks at a time (default 1)
#   PYTHON=python3.12 sh run_all.sh   choose the interpreter (default python3; needs Python >= 3.9)
#   SKIP_SLOW=1 sh run_all.sh         skip the two ~1-minute Part II bridge controls
#
# Steps: input manifests, Part I certificates and selftest, Part II families, selftest and the five canonical
# points (pure-Python kernel), must-fail controls of both parts, table bodies, comparison with the committed
# reference results and tables.  Everything is written under OUTDIR; the exit status is 0 only if every step
# succeeded and every control was rejected for the expected reason.
set -u
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python3}
OUT=${1:-out}
JOBS=${JOBS:-1}
mkdir -p "$OUT/partI" "$OUT/partII" "$OUT/logs"
LOG="$OUT/run_all.log"
: > "$LOG"
status=0

say() { echo "$*" | tee -a "$LOG"; }
step() {  # step NAME CMD...: run, log to OUT/logs/NAME.log, record exit status and wall time
    name=$1; shift
    t0=$(date +%s)
    "$@" > "$OUT/logs/$name.log" 2>&1
    rc=$?
    t1=$(date +%s)
    say "[$name] exit $rc, $((t1 - t0)) s: $(tail -n 1 "$OUT/logs/$name.log")"
    [ "$rc" -eq 0 ] || status=1
}

"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else "Python >= 3.9 is required")' || exit 1
say "run_all.sh: $("$PY" -c 'import platform, sys; print("Python", sys.version.split()[0], "on", platform.system())'), JOBS=$JOBS, OUT=$OUT"

step manifest_partI "$PY" tools/manifest.py verify partI/inputs/SHA256SUMS.txt
step manifest_partII "$PY" tools/manifest.py verify partII/inputs/SHA256SUMS.txt
step partI_check "$PY" partI/check_partI.py --manifest partI/inputs/SHA256SUMS.txt --out-dir "$OUT/partI"
step partI_selftest "$PY" partI/selftest_partI.py
for n in 3 4 5 7 8; do
    step "partII_family_n$n" "$PY" partII/check_family_stdlib.py --n "$n" --family "partII/inputs/family_n$n.json" \
        --out "$OUT/partII/family_n$n.json"
done
step partII_selftest "$PY" partII/selftest_stdlib.py --family partII/inputs/family_n8.json --n 8

# the five canonical Part II points (n = 3 is the longest, so it starts first)
t0=$(date +%s)
printf '3\n4\n5\n7\n8\n' | xargs -P "$JOBS" -I{} sh -c \
    '"$0" partII/check.py --n "$1" --point "partII/inputs/point_n$1.json" --threads 1 --kernel python \
      --out "$2/partII/check_n$1.json" > "$2/logs/partII_check_n$1.log" 2>&1' "$PY" {} "$OUT"
t1=$(date +%s)
for n in 3 4 5 7 8; do
    if grep -q '"verdict": "ALL CHECKS PASSED"' "$OUT/logs/partII_check_n$n.log"; then
        say "[partII_check_n$n] ALL CHECKS PASSED: $(grep -o '"wall_s": [0-9.]*' "$OUT/partII/check_n$n.json" | head -n 1)"
    else
        say "[partII_check_n$n] FAILED: $(tail -n 1 "$OUT/logs/partII_check_n$n.log")"
        status=1
    fi
done
say "[partII_points] $((t1 - t0)) s wall for the five points with JOBS=$JOBS"

if [ "${SKIP_SLOW:-0}" = 1 ]; then
    step controls "$PY" tools/run_controls.py --out "$OUT/controls" --jobs "$JOBS" --skip-slow
else
    step controls "$PY" tools/run_controls.py --out "$OUT/controls" --jobs "$JOBS"
fi
step tables "$PY" tables/make_tables.py --results "$OUT" --out "$OUT/tables"

if [ -d results/partI ]; then
    step compare_reference "$PY" tools/compare_results.py --ref results --new "$OUT"
    for f in partI_thresholds.tex partI_oneshot.tex partII_points.tex; do
        if cmp -s "tables/$f" "$OUT/tables/$f"; then say "[tables] $f identical to the committed tables/$f"
        else say "[tables] $f DIFFERS from the committed tables/$f"; status=1; fi
    done
fi

if [ "$status" -eq 0 ]; then say "ALL VERIFICATIONS PASSED"; else say "SOME STEP FAILED (see $OUT/logs)"; fi
exit "$status"
