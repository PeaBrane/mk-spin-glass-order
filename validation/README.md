# Independent cross-checks (not part of the proof)

The proofs in the paper rest on the checkers in `partI/` and `partII/`. This directory holds two further
implementations, written independently of those checkers, and the records of their runs. They are cross-checks
reported in Appendix B.5 of the paper; nothing in the paper depends on them.

## `partI_independent/`: two further Part I verifiers

- `qcert.py` (standard library only) and `arbcert.py` (python-flint Arb, 400 bits) were written from the
  statements of Proposition A, Lemma T and the transfer theorem alone, without reading `partI/`. They decide the
  same hypotheses as `partI/check_partI.py`: `w_k < 1` at every level, (T0)–(T2) of Lemma T, and the order and
  imbalance inequalities where a certificate claims them. `qcert.py` works in exact fractions with its own series
  enclosures of π, exp and ln 2 (outward rounding to 256-bit mantissas).
- Both accept all 55 frozen inputs of `partI/inputs/` (the committed files): `qcert_all55.jsonl`,
  `arb_all55.jsonl`, made with the commands below (Python 3.12, python-flint 0.9.0). Their margins agree to
  1.1e-16. `partI_row_margins.tsv` lists the floors of the pole, order and imbalance bounds per row.
- `qcontrols.py` applies single mutations (β − 0.1, n − 1, a dropped last level, and others) to every input;
  `qcontrols.jsonl` records that every such mutation of β, n and the last level is rejected.

Rerun from the repository root:

```sh
python3 validation/partI_independent/qcert.py partI/inputs/*/*.json > qcert_all55.jsonl
python3 validation/partI_independent/arbcert.py partI/inputs/*/*.json > arb_all55.jsonl   # needs python-flint
PYTHONPATH=validation/partI_independent python3 validation/partI_independent/qcontrols.py partI/inputs > qcontrols.jsonl
```

## `partII_rc/`: a second Part II implementation

`code/` is an earlier clean-room implementation of the Part II certificate (its own search, bridge and exact
checker; numpy, numba and python-flint are required). It was written from an earlier write-up of Section 6 of the
paper. Before this repository existed it certified n = 3, 4 and 8. The records here are its runs at the two
remaining points, made without reading `partII/`:

- **n = 5, β = 5/6** (T = 6/5): family (λ, A, ε₀) = (123/100, 12, 1/300); bridge certified at J = 25; the unmodified
  checker prints `ALL CHECKS PASSED` (`results/chk/chk_Q5_n5_b5_6.json`).
- **n = 7, β = 4/7** (T = 7/4): family (349/250, 12, 1/300); bridge certified at J = 18. The unmodified checker stops
  with `intermediate accumulator not saved` (`results/chk/chk_Q7_n7_b4_7.log`): it only verifies addition plans whose
  intermediate sums are powers of two or n, and the plan for 7 is 1+1→2, 2+2→4, 4+2→6, 6+1→7.
  `code_ext/rc_accwitness.py` replays each level's plan with the unchanged convolution code and saves the 6-fold
  intermediate law as an untrusted witness; `code_ext/rc_check_acc.py` (the change, 11 added lines, is
  `code_ext/rc_check_acc.diff`) validates each witness as a staircase law and checks it with the checker's own exact
  convolution test, then uses it as the source of the next step. It prints `ALL CHECKS PASSED`
  (`results/chk_ext/chk_Q7_n7_b4_7.json`). Controls: the extended checker reproduces the unmodified checker's
  numbers on n = 5 (`chk_Q5_n5_b5_6_control.json`), and a corrupted level-0 witness is rejected
  (`chk_Q7_negcontrol.log`).
- All family, bridge and witness outputs were hashed before any check (`results/logs/SHA256SUMS_*`); the hashes
  were verified again after checking. The outputs themselves (about 8 MB) are not included; they are regenerated
  deterministically by

```sh
cd validation/partII_rc
python code/rc_family.py --n 5 --lam 123/100 --A 12 --eps0 1/300 --kappa 0.945 --Npad 2560 --out fam/fam_n5
python code/rc_bridge.py --n 5 --beta 5/6 --e0 9 --N 12000 --family fam/fam_n5 --threads 4 --maxlev 400 --out runs/Q5_n5_b5_6
python code/rc_check.py --run runs/Q5_n5_b5_6 --threads 2 --out chk_Q5_n5_b5_6.json
python code/rc_family.py --n 7 --lam 349/250 --A 12 --eps0 1/300 --kappa 0.945 --Npad 3200 --out fam/fam_n7
python code/rc_bridge.py --n 7 --beta 4/7 --e0 9 --N 12000 --family fam/fam_n7 --threads 4 --maxlev 400 --out runs/Q7_n7_b4_7
PYTHONPATH=code python code_ext/rc_accwitness.py --run runs/Q7_n7_b4_7 --acc_dir accw/Q7_n7_b4_7
python code_ext/rc_check_acc.py --run runs/Q7_n7_b4_7 --acc_dir accw/Q7_n7_b4_7 --threads 2 --out chk_Q7_n7_b4_7.json
```

(numpy, numba and python-flint must be installed). The recorded runs used Python 3.13.13, numpy 2.5.3,
numba 0.67.0, python-flint 0.9.0 and mpmath 1.3.0.

**Edits after the recorded runs (comments only).** The code in `code/` is the code of the recorded runs; its
hashes at the time of the runs are in `results/logs/SHA256_code.txt`. Afterwards, the docstrings of two files,
which named files that are not part of this repository, were reworded; no code line changed. The other three
files are unchanged.

| file | SHA-256 recorded for the runs | SHA-256 now |
|---|---|---|
| `code/rc_core.py` | `cd29766a1b68ed41717144aa7a2dfff7a28cb03a5e31f83d1741ace358afdcf4` | `d5dea3054b89b71bb17d08d7988551a329e02f797e905c78b4f8269df790c168` |
| `code/rc_family.py` | `907680de6c91e3543b01817fcc7181ae238e3e23058e4560aa0b29288940c210` | `71d592fa3329109df9b41c951938df8ef9952d1b76ed75f05f3453896fb1b7dc` |

Likewise, in `code_ext/` the first docstring line of `rc_accwitness.py`, one comment and one `--help` string of
`rc_check_acc.py`, and the header of `rc_check_acc.diff` (file names without timestamps) were reworded; none of
these changes affects a computation. In the run records the Python build string was shortened to the version
number.
