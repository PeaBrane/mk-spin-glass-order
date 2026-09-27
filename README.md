# mk-spin-glass-order: certificates for spin-glass order on Migdal–Kadanoff diamond lattices

This repository re-verifies every certified number in the paper. Only the Python standard library is needed
(Python >= 3.9), and a full run takes minutes. It contains:

- **Part I** (analytic route, Gaussian couplings, n >= 4 branches): the one-level anti-concentration map
  (Prop. A), the geometric tail (Lemma T) and the constants of the Transfer Theorem. It checks the
  hand-checkable one-shot pole certificates and the multi-level certificates behind the threshold table.
- **Part II** (computer-assisted, near T_c, n = 3, 4, 5, 7, 8): the staircase-law bridge, the scale-covariant family,
  the entry condition and the constants of Theorem B (ii)–(iv) (checker `mkcert`; `partII/CONDITIONS.md` maps its
  labels to the paper's).
- `tables/make_tables.py`, which writes the table bodies used by the manuscript from result files only.
- Must-fail controls for both parts. `run_all.sh` runs everything.
- `validation/`: independent cross-checks and their records (not part of the proof; see `validation/README.md`).

The model: the b = 2 diamond hierarchical lattice with n branches, and couplings J_e iid N(0, 1). The effective
coupling obeys

    ξ_{k+1} =_d Σ_{i=1}^{n} atanh(tanh a_i tanh b_i),    a_i, b_i iid ~ ξ_k,    ξ_0 = βJ.

## 1. What is certified

### 1.1 Part I (`partI/`, inputs in `partI/inputs/`)

H_k(r, w) means that there is an event E_k of the couplings of a scale-k sub-diamond with P(E_k^c) <= w and
P(ξ_k ∈ dx, E_k) <= r dx. The checker verifies, in exact rational arithmetic, the hypotheses of the following two
results.

- **Prop. A.** "Assume H_k(r, w) with w < 1. For any L, R > 0 put
  ρ := r/(1 − w), p := 2w + (2ρR)² + 4ρL, θ := (1 + e^{−2L}) / tanh R,
  Γ := θ · Σ_{g=1}^{n} C(n,g) κ_g p^{n−g}. Then H_{k+1}(rΓ, p^n) holds."
  Here κ_g = p₂^{*g}(0) with p₂(x) = 2(1 − 2|x|)₊, so κ_1..κ_4 = 2, 4/3, 11/10, 302/315. (The paper writes ϖ
  and ϑ for p and θ.)
- **Lemma T.** "Suppose H_K(r_K, w_K) and there are A, γ, L, R > 0 with (T0) w_K ≤ A r_K², A r_K² < 1,
  (T1) Γ(r_K, A r_K²) ≤ γ < 1 and (T2) p(r_K, A r_K²)^n ≤ A γ² r_K². Then H_k(r_K γ^{k−K}, A r_K² γ^{2(k−K)})
  for all k ≥ K."

A certificate (`partI/inputs/*/*.json`) consists of β, levels (L_k, R_k) for k < K, and a tail (A, γ, L, R). The
checker (`partI/propa.py`) performs these steps:

1. It starts from H_0(r_0, 0) with r_0 ≥ φ/β and φ = 1/√(2π), the maximal density of N(0, 1).
2. At every level it checks w_k < 1 and L_k, R_k > 0, and sets r_{k+1} ≥ r_k Γ and w_{k+1} ≥ p^n (Prop. A).
3. At level K it checks (T0), (T1) and (T2) exactly as displayed above.
4. It accumulates the Transfer-Theorem constants of the analytic feeder
   (F_k = w_k + 2r_k t, Q_k = w_k + 4r_k t):

       ℓ_s = Σ_{k≥0} (w_k/2 + 2r_k)                                          (spin-only rate, Lemma S)
       ℓ_b = Σ_{k≥0} [(3/4)w_k + (1 + 2 ln 2) r_k + w_k² + w_k r_k + r_k²/2]   (blue rate, Lemma B′)
       S = sup_k (w_k + 2r_k),   P = sup_k (w_k + 2 ln 2 · r_k),

   with the tail k ≥ K summed in closed form.
5. It verifies the certificate's claims:
   - **pole**: the certificate closes. Then E⟨σ_Aσ_B⟩² ≥ 1 − S for every N, and E⟨σ_Aσ_B⟩² → 1.
   - **order**: 1 − 2ℓ_s − S > 0. Then the site overlap satisfies E Q² ≥ (1 − 2ℓ_s − S)₊² > 0 for every N, and the
     fixed-pole EA order satisfies E⟨σ_x⟩² ≥ 1 − 2ℓ_s.
   - **imbalance**: (1 − 2ℓ_b)₊² − P > 0. Then the blue-cluster imbalance satisfies
     E D² ≥ (1 − 2ℓ_b)₊² − P > 0 for every N.
   - **bounds**: the certificate closes. The four lower bounds are reported.

A certificate at β also holds at every β′ ≥ β with the same constants, because H_0(φ/β, 0) implies H_0(φ/β′, 0)
and H is monotone. Hence a threshold is certified for all lower temperatures.

**Status of the mathematics** (audited separately, not re-done by the code):
- Prop. A holds with one repaired proof step (the density factor r/π instead of ρ in the last line). The repair
  leaves Γ and p unchanged.
- Lemma T and the Transfer Theorem hold. Item 6 of the Transfer Theorem has a local repair (Jensen is applied to
  the complement of C(A) ∪ C(B)) that leaves the statement unchanged.
- The code decides only the displayed inequalities.

### 1.2 Part II (`partII/`, inputs in `partII/inputs/`)

For each (n, T*) = (3, 4/15), (4, 6/7), (5, 6/5), (7, 7/4), (8, 2), the frozen inputs are a family file
(λ, A, ε₀, the shape X_c and witnesses G_1..G_n) and a point file (β = 1/T*, grid, branch parameters, coarsening
schedule, J, and the family's SHA-256). `partII/check.py` rebuilds ν_0, …, ν_J deterministically and verifies
these conditions:

- **(S)** every ν_j is symmetric unimodal.
- **(B0)** "CDF_{ν₀}(t) ≥ erf(t/(β√2)) for all t".
- **Coarsening, (Bj.1)–(Bj.3).** Branch map, n-fold sum and the choice of ν_{j+1} at every level.
- **(C0), (C1), (C2).** (C1) is "s(t) := F_c(t/λ) − G_n(t) ≥ 0"; (C2) is
  "s(t) + ε₀(A/λ)(1 − F_c(t/λ)) ≥ ε₀ [ n g_n(t) + Σ_{k=1..n} C(n,k)(2A)^k ε₀^{k−1}(Ht_{n−k}(t) − G_n(t)) ]".
  Also checked: λ > 1, 2Aε₀ ≤ 1, and that X_c is atomless with support in [0, x_max].
- **(E) entry.** "CDF_{ν_J}(t) ≤ α₀ + (1 − α₀) F_c(t/(L₀h_c)) on [0, L₀x_max]", with L₀ = C0_UP/ε₀ ≥ c₀/ε₀ and
  α₀ = A·C0_LO/L₀ ≤ A c₀/L₀. Here F_c is evaluated in index units of the grid h_c = x_max/N_x and c₀ = (ln 2)/2.
  The rational bounds C0_LO < (ln 2)/2 < C0_UP are themselves verified.

Every condition is checked on the continuum, not only at grid points. `partII/CONDITIONS.md` maps each condition to
its code and gives the reason a finite check suffices. Together with the mathematical lemmas of the paper (Lemma U,
Lemma M, (E1), Theorems 1A and 1B), a passing check gives:

- **Theorem 1 at every β′ ≥ β.** For i ≥ 0, P(|K_{J+i}| ≤ u) ≤ α_i + (1 − α_i) F_c(u/L_i) ≤ (A c₀ + f_c u)/(L₀λ^i).
  Hence |K_p| → ∞ in probability.
- **Constants for Theorems 2 and 3.** These are exact rational lower bounds of
  κ_Q^b = Σ_j (2n−1)(2n)^{−j}(1 − 2T_j)₊ and κ_C = Σ_j (2n−1)(2n)^{−j}(1 − T_j)₊, where T_j = Σ_{l≥j} φ_l is
  evaluated bin by bin from ν_0..ν_J and from the family tail. Then liminf E⟨R₁₂²⟩ ≥ (κ_Q^b)² and
  liminf E[Imb²] ≥ κ_C².

The κ_Q of the table is the blue-route constant κ_Q^b. It is a valid constant in Theorem B(ii) because the spin
rates are bounded by the blue rates (paper, Corollary "Level-weighted limits").

### 1.3 Which result file gives which table entry

| table body (`tables/`) | rows | columns | source |
|---|---|---|---|
| `partI_thresholds.tex` | n = 4, 5, 6, 7, 8, 12, 16, 32, 64 | n & β_pole & β_order & β_imb | `results/partI/thresholds__*.result.json` (`result.beta`, claim verified) |
| `partI_oneshot.tex` | the same n | n & β & L & R & A & γ | `results/partI/oneshot__oneshot_n*.result.json` |
| `partII_points.tex` | n = 3, 4, 5, 7, 8 | n & T* & β* & J & log10 κ_Q & log10 κ_C | `results/partII/check_n*.json` |

`tables/table_sources.md` lists every printed number with its file and field.

How the numbers are printed:
- Thresholds are the exact β of a verified certificate, printed with one decimal and rounded up. All of them lie
  on the 0.1 grid, so rounding does not change them.
- One-shot parameters are exact fractions.
- The log10 values are the checker's floors to 3 decimals, taken from the lower endpoint of an interval enclosure.

What "smallest" means for a threshold:
- The next grid point below each threshold (β − 0.1) was not certified by the float search. This is a search
  outcome, not a proof that no certificate exists there.
- What is checked is weaker and checkable: the frozen parameters themselves fail at β − 0.1. See the
  `beta_minus_0.1_*` controls.

## 2. How to reproduce

```sh
git clone https://github.com/PeaBrane/mk-spin-glass-order && cd mk-spin-glass-order
sh run_all.sh                 # everything, one process at a time; writes out/
JOBS=5 sh run_all.sh          # the same with up to 5 checks in parallel
```

`run_all.sh` needs only `python3` (>= 3.9) and a POSIX shell. It performs these steps:

1. Verifies the input manifests (`partI/inputs/SHA256SUMS.txt`, `partII/inputs/SHA256SUMS.txt`).
2. Checks the 55 Part I certificates.
3. Runs both standard-library selftests.
4. Checks the five Part II families.
5. Checks the five Part II points with the pure-Python kernel.
6. Runs the 46 must-fail controls.
7. Writes `out/tables/*.tex`.
8. Compares every result with the committed reference results (`results/`) and every table body with
   `tables/*.tex`.

The last line is `ALL VERIFICATIONS PASSED`, and the exit status is 0.

Individual commands:

```sh
python3 partI/check_partI.py --manifest partI/inputs/SHA256SUMS.txt --out-dir out/partI      # Part I, seconds
python3 partI/check_partI.py --input partI/inputs/oneshot/oneshot_n4.json --n 4 --out out/n4.json
python3 partII/check_family_stdlib.py --n 4 --family partII/inputs/family_n4.json --out out/fam4.json  # ~1 s
python3 partII/check.py --n 4 --point partII/inputs/point_n4.json --threads 1 --kernel python --out out/n4.json
python3 tools/run_controls.py --out out/controls --jobs 4
python3 tables/make_tables.py --results results --out tables                                # regenerate the tables
```

`check.py` exits with status 2 and prints `CHECK FAILED: ...` if any inequality fails; `check_partI.py` does the same.

**Run times.** These were measured on the reference run (`results/run_all.log`): Python 3.12 on a Linux machine,
one core per check, `JOBS=6`.

- Part I: all 55 certificates in 3.8 s. Both selftests together take about 4 s.
- Part II families: 0.2–0.7 s each.
- Part II points: n = 3, 4, 5, 7, 8 in 185, 65, 48, 42 and 39 s (378 s in total), at about 40 MB peak memory each.
- Controls: 36 s. The longest control is the J − 1 control at 36 s.
- The whole of `run_all.sh`: about 4 minutes of wall time with `JOBS=6`.
- A fresh copy of this tree, including `results/`, run with `JOBS=1` took about 430 s (7 minutes)
  (`results/run_all_serial.log`). It reproduced all 65 reference result files and the three table bodies exactly.

**Python versions.**
- The reference run used Python 3.12.
- Earlier runs of the same checks on the same certificate values passed with Python 3.10 (the whole of
  `run_all.sh`) and with Python 3.13 (the Part I checks, both selftests and the n = 8 family check).
- The code uses nothing newer than Python 3.9; its syntax was checked with `ast.parse(..., feature_version=(3, 9))`.
- An earlier version of the Part II checker, from which `partII/` derives, ran with Python 3.9. It checked
  n = 8 in 57 s at 36 MB peak memory.

**Optional accelerators and tools.** None of these is needed for any certified number.
- `partII/check.py --kernel numpy|numba` evaluates the branch sums with numpy or numba. They give the same
  integers, faster.
- `partII/optional/` holds the float search that produced the Part II inputs (`search.py`, `shape.py`,
  `freeze_points.py`; numpy) and the full selftest (numpy, numba, scipy, and python-flint for an Arb cross-check).
  `search.py` also writes search diagnostics (`search_info`: timings, code hashes, environment); they are not part
  of a certificate, no check reads them, and the committed inputs omit them.
- `partI/optional/search_partI.py` is the float search for Part I certificates (scipy). Every certificate it writes
  is re-verified by the standard-library checker.
- `partI/optional/check_iv_mpmath.py` is an independent second implementation of the Part I check in mpmath
  interval arithmetic, used before the inputs were frozen (unchanged apart from its docstring and one comment). It
  reads the frozen inputs directly.

## 3. Arithmetic model

- **No float decides anything.** Every decision is an exact comparison of Python integers or `fractions.Fraction`s.
- **Transcendental numbers** (exp, log, erf, π, √) come from `partII/mkcert/ivdec.py`. These are interval
  enclosures in `decimal` at 64 significant digits. Every operation is done twice, rounding toward −∞ and toward
  +∞. The series have explicit remainder bounds: Taylor series for exp, atanh (for log) and erf, and Machin's
  formula for π. Square roots are verified by squaring. Decisions use the exact rational endpoints.
- **Part II units.** Probabilities are integers in units of 2^−48. Transcendental tables are rounded outward to
  integers, at scale 2^(R+e) for δ, 2^48 for erf, and 2^96 for ψ and e^{−4s}.
- **Part I.**
  - The quantities φ, e^{−2L}, e^{−2R} and ln 2 are replaced by one-sided rational bounds.
  - The carried bounds r_k and w_k, and the intermediate p and Horner sums, are rounded up to 64 significant
    digits. This is valid because every map involved is nondecreasing in them.
  - Everything else is exact.
- **Where floats appear.** Floats appear only in the choice of a tangent point, of an argument reduction or of a
  working precision. Each choice is re-verified exactly or cannot affect validity. The list is in
  `partII/CONDITIONS.md` §6.
- **Inputs.** Rationals are strings (`"7/6"`; Part I also accepts exact decimals such as `"2.478"`). Integers are
  JSON integers. JSON floats, booleans, NaN and infinities are rejected.

## 4. Layout

```
README.md, run_all.sh
partI/   propa.py (trusted: Prop. A, Lemma T, constants), check_partI.py (driver), selftest_partI.py,
         inputs/{oneshot,thresholds,bounds}/*.json + SHA256SUMS.txt, controls/, optional/
partII/  check.py, check_family_stdlib.py (drivers), mkcert/ (checker package), selftest_stdlib.py, CONDITIONS.md,
         inputs/{family,point}_n{3,4,5,7,8}.json + SHA256SUMS.txt, controls/, optional/
tables/  make_tables.py and the committed table bodies partI_thresholds.tex, partI_oneshot.tex, partII_points.tex,
         table_sources.md
tools/   manifest.py, run_controls.py, compare_results.py
results/ the reference run (result files, logs, control outcomes) from which tables/ was generated
validation/ independent cross-checks and their records (not part of the proof)
```

`results/` holds the reference run of this exact tree:
- `partI/*.result.json` and `partII/{check,family}_n*.json`.
- `controls/controls_summary.json`.
- `logs/` and `run_all.log`, plus `run_all_serial.log` from a second, serial run of a fresh copy of this tree
  that compared its results with `results/`.

`tables/*.tex` were generated from `results/` with `python3 tables/make_tables.py --results results --out tables`.
A fresh `run_all.sh` compares its own results with `results/` field by field, ignoring timings and environment,
and compares its tables byte for byte.

Part I inputs come in three groups:
- `oneshot/` holds the nine hand-checkable one-shot pole certificates.
- `thresholds/` holds the certificates at the three thresholds for each n. Where the order and imbalance
  thresholds coincide, one file serves both.
- `bounds/` holds 24 multi-level certificates at reference β values (for example n = 4, β = 530:
  E⟨σ_x⟩² ≥ 0.658577 with fixed poles). The paper may quote these. They are checked by `run_all.sh`, but no table
  body is made from them.

Each Part I input carries a one-line `description` of how it was found; no check reads it.

## 5. Must-fail controls

`tools/run_controls.py` builds and runs 46 invalid inputs (32 for Part I, 14 for Part II), each changing one
thing. Every one must be rejected with exit status 2 and the expected reason. The outcomes of the reference run are
in `results/controls/controls_summary.json`.

Part I controls:
- γ − 10⁻⁵, γ × 0.999, and A/1000.
- Every threshold certificate at β − 0.1 (22 files).
- A dropped last level.
- Wrong n: in the file (4 → 3 and 5 → 4) and on the command line.
- A pole certificate claiming order.
- β written as a JSON float.
- A tampered manifest hash.

Part II controls:
- J − 1.
- β lowered by 0.1 (n = 8: β = 2/5).
- A wrong family hash.
- Wrong n, on the command line and in the file.
- A JSON float for J, and a JSON float mass.
- λ = 2, and ε₀ × 10.
- A (C1) violation of one unit at t = 0.
- A per-law hash mismatch.
- Two tampered n-fold plans.
- A tampered manifest hash.

## 6. Changes relative to the audited version of the checker

An independent audit examined an earlier version of the Part II checker (paper, Appendix B.5). The published
checker differs from it as follows.

- The labels (i1, i2, k) of the n-fold convolution plan are now verified by trusted code
  (`exact.check_nfold_plan`). It checks that k = i1 + i2, that both inputs exist, that no label is reused, and
  that the last label is n. ν_{j+1} is taken from that last step. Previously the labels came unverified from
  `construct.nfold_plan`.
- Integers must be JSON integers, and nothing is coerced with `int()`. Rationals must be strings. NaN and
  infinities are rejected, and each law's recorded `sha256` is verified.
- The stopping rule of the erf series is an exact rational test. All uses of floats are documented.
- `partII/CONDITIONS.md` gives the full condition-to-code map, including `branch.fine_bounds`,
  `constants.verify_rational_constants` and the X_c checks.
- 2Aε₀ ≤ 1 is tested explicitly. It was already implied by (C1) and (C2) at t = 0.

No certified value changed. Every certificate value in the ten Part II inputs is identical to the inputs checked
by the audited version (the family hashes in the point files changed only because the family files no longer
carry search diagnostics), and the chain fingerprints, J, all margins and all constants equal that version's
recorded verification.

## 7. Scope and limits

- **What the code checks.** The code checks the inequalities listed above. The mathematical lemmas that turn them
  into theorems are proved in the paper and were audited separately. These are Prop. A, Lemma T, Lemmas S and B′
  and the Transfer Theorem (Part I), and Lemma U, Lemma M, (E1), Theorems 1A and 1B and the chain lemmas (Part II).
- **Trusted computing base.** The trusted base is CPython's `int`, `fractions` and `decimal`, plus the trusted
  modules listed in `partII/CONDITIONS.md` §5 and `partI/propa.py`. With `--kernel numpy|numba` it also includes
  numpy or numba.
- **Sharper Part II points are not included.** Points with a finer grid and larger κ bounds were also found; the
  paper does not use them.
