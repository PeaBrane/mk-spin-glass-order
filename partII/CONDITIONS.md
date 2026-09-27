# Part II: condition-to-code map of the checker `mkcert`

This file lists every condition that a Part II certificate must satisfy and the code that decides it. The
conditions are those of the certificate definitions in Section 6 of the paper.

Concordance with the paper's names (this file keeps the labels used in the code):

| here and in the code | in the paper |
|---|---|
| Def. 4.1 (bridge), Def. 4.4 (family), Def. 4.5 (certificate) | Definitions "bridge certificate", "family certificate" and "certificate" (Section 6) |
| Theorem 1B, Theorem 1A, Theorem 1 | Theorems "Bridge", "Invariance of the family" and "Coupling growth" (Section 6) |
| Theorems 2 and 3 | Theorem B (ii)–(iv) |
| (E) | (EN) |
| (Bj.1) | checked by (V2) (branch map) of the paper's Appendix B |
| (Bj.2)–(Bj.3) | checked by (V3) (n-fold sum) of the paper's Appendix B |
| F_c in index units, F_c(t/(L_0 h_c)) with h_c = x_max/N_x | F_c(t/L_0) in the variable t |

Notation. ν_j is the staircase law at bridge level j (grid h = 2^−e, N bins). F_j is its CDF. X_c is the family
shape with CDF F_c. G_1, …, G_n are the family witness laws with right derivatives g_m. f(a, b) =
atanh(tanh a tanh b), and ⪯ is "more peaked", i.e. a CDF bound from above between symmetric unimodal (s.u.)
magnitude laws. Every check raises `CertificateError`, and `check.py` then exits with status 2.

## 1. Bridge (Theorem 1B): `mkcert/bridge.py` in CHECK mode, driven by `check.py`

| condition | statement | deciding code | why a finite check covers the continuum |
|---|---|---|---|
| (S) | every ν_j is an s.u. magnitude law | `exact.validate` on ν_0, on every coarsened law, on every branch output Y and on every conv output (`bridge.run`) | nonincreasing integer masses summing to exactly 2^48 give a concave CDF |
| (B0) | F_0(t) ≥ erf(t/(β√2)) for all t ≥ 0 | `gauss.check_gauss` (erf, exp, sqrt, π from `ivdec`) | on each bin with C_k < 1, erf(·/(β√2)) − ℓ_k is concave. Its tangent line at a dyadic point is ≤ 0 at both bin ends (upper interval endpoints). Bins with C_k = 1 need nothing |
| coarsening | the coarsened law ⪯ the old one (grid h → 2h) | `exact.check_coarsen` | both CDFs are linear between old knots |
| (Bj.1) | Y ⪯ law of f(A, B), A, B iid ~ ν_j | `branch.fine_bounds` (trusted) builds integers U_i ≥ 2^48 P(f(A,B) ≤ ih/s) from the exact identity g(u,b) = u + δ(b−u) − δ(b+u), with outward-rounded δ tables from `ivdec`. Then `exact.check_branch_output` verifies CDF_Y(u_i) ≥ U_{i+1} | P(f ≤ t) is nondecreasing in t, so P(f ≤ t) ≤ U_{i+1} ≤ CDF_Y(u_i) ≤ CDF_Y(t) on [u_i, u_{i+1}]. CDF_Y = 1 beyond Nh |
| (Bj.2) | each partial sum of the binary-doubling plan ⪯ the sum of its two inputs | `exact.check_nfold_plan` checks the labels (k = i1 + i2, both inputs available, no label reused, last label n). `exact.check_conv` checks every step | per-bin quadratic slack checked at σ = 0, at σ = 1 and at the vertex when it lies in (0, 1). CDF = 1 beyond Nh |
| (Bj.3) | ν_{j+1} ⪯ the n-branch sum | `bridge.run`: ν_{j+1} is the output of the plan's last step, whose label is required to be n (equality) | — |
| schedule | the frozen coarsening counts and J are used | `check.py` reads them as JSON integers. `bridge.run` requires len(schedule) = J | — |
| (E) | F_J(t) ≤ α₀′ + (1 − α₀′) F_c(t/L₀) for all t, with L₀ ≥ c0/ε₀ and α₀′ ≤ A c0/L₀ | `exact.check_entry` with L₀ = C0_UP/ε₀ and α₀ = A·C0_LO/L₀ (`check.py`) | both sides are piecewise linear, so the union of their knots on [0, L₀x_max] suffices. The right side equals 1 beyond |
| c0 | C0_LO < (ln 2)/2 < C0_UP, and (ln 2)/2 + 1/4 < LN2H_Q_UP | `constants.verify_rational_constants` (ln 2 = 2 atanh(1/3) in `ivdec`), called by `check.py` before L₀ is formed | — |

## 2. Family (Theorem 1A): `mkcert/family.py` and `exact.check_family`

| condition | statement | deciding code | coverage |
|---|---|---|---|
| (F1) | X_c and G_1..G_n are staircase laws. X_c is atomless with support in [0, x_max] | `exact.validate`, plus `check_family`: `xa == 0` and zero mass beyond N_x | — |
| parameters | λ > 1, A > 0, ε₀ > 0, 2Aε₀ ≤ 1 | `check_family`. The last test is explicit; it is also implied by (C1) and (C2) at t = 0 | — |
| (C0) | G_1 ⪯ min2(X_c), and G_{m+1} ⪯ G_m ⊕ G_1 | `exact.check_min2`, `exact.check_conv` | exact quadratic slack per bin |
| (C1) | s(t) := F_c(t/λ) − G_n(t) ≥ 0 on [0, λx_max] | `check_family` | union of the knots {integers} ∪ {λk} |
| (C2) | s(t) + ε₀(A/λ)(1 − F_c(t/λ)) ≥ ε₀[n g_n(t) + Σ_{k=1..n} C(n,k)(2A)^k ε₀^{k−1}(Ht_{n−k}(t) − G_n(t))], with H_m = G_m + nε₀g_m, Ht_m = max_{m′≥m} H_{m′} ≥ Ĥ_m and Ht_0 = 1 | `check_family` | the margin is concave on each elementary interval. Both ends are evaluated with the interval's own slopes, the right end as a left limit |

## 3. Binding and inputs

| condition | deciding code |
|---|---|
| one n everywhere | `check.py`: `--n` = n of the point file = n of the family file (`load_family`) |
| the family in (E) is the certified one | `check.py`: the family SHA-256 equals the value recorded in the point file. The same in-memory object is verified and then used for (E) and for the constants |
| inputs are what they say | `common.load_json` rejects NaN and ±Infinity. `common.as_int` accepts only JSON integers (no `int()` coercion). `common.frac` accepts only 'p/q' or integer strings. `common.law_from_json` requires integer masses and verifies each law's recorded `sha256` |
| input files | `tools/manifest.py verify partII/inputs/SHA256SUMS.txt` (run by `run_all.sh`) |

## 4. Constants of Theorems 2 and 3 (not needed for Theorem 1)

| quantity | code |
|---|---|
| φ_l ≤ E ψ(V/2) + E e^{−4 max(V,V′)} for l − 1 ≤ J, bin by bin | `constants.phi_bridge`. ψ, Ψ and e^{−4s} values are ceilings at scale 2^96 of `ivdec` upper endpoints |
| family tail Σ_{i≥1}(a1/L_i + a2/L_i²) | `constants.family_tail_coeffs`, `constants.family_tail` (exact rationals, with C0_UP and LN2H_Q_UP) |
| κ_Q^b = Σ_j (2n−1)(2n)^{−j}(1 − 2T_j)_+, κ_C = Σ_j (2n−1)(2n)^{−j}(1 − T_j)_+ | `constants.kappas`. The sums are exact and truncated by dropping nonnegative terms |
| printed floors | `exact.floor_sci`, and `check.log10_floor`, which takes the lower endpoint of an `ivdec` log enclosure |

The κ_Q computed here is the blue-route constant κ_Q^b. It is valid in Theorem B(ii) because the spin rates are
bounded by the blue rates, so κ^s ≥ Σ_j (2n−1)(2n)^{−j}(1 − 2T^b_j)₊ (paper, Corollary "Level-weighted limits"). The q-chain constant κ_Q, which is larger, is not implemented.

## 5. Trusted computing base

- CPython's `int`, `fractions` and `decimal` with directed rounding.
- `mkcert/exact.py`, `family.py`, `branch.py` (the bound and the pure-Python kernel), `gauss.py`, `constants.py` and
  `ivdec.py`.
- `common.py` (parsing and hashing).
- `bridge.py`, which sequences CHECK mode: the plan verification, the order of the checks, and the choice of
  ν_{j+1}.
- The drivers `check.py` and `check_family_stdlib.py`.
- The optional `numpy` and `numba` kernels in `branch.py` are used only with `--kernel numpy|numba`. They compute the
  same integers as the pure-Python kernel, and the reference run does not use them.

`construct.py` is untrusted. Every law it outputs is re-verified, and its one non-law output, `nfold_plan`, is
verified by `exact.check_nfold_plan`. Everything in `optional/` is untrusted and unused by the checks.

## 6. Where floats appear

No float decides an inequality.
- **`gauss.check_gauss`.** A float chooses the tangent point, which is then rounded to a dyadic rational. Any point
  of the bin is valid.
- **`ivdec._exp_point`.** A float picks the halving count. The reduced argument is then tested exactly, and a failed
  test raises an error.
- **`ivdec._log_point`.** A float guesses the binary exponent. It is corrected by exact Decimal comparisons.
- **`ivdec._erf_point`.** A float selects the branch (both branches are valid for every x > 0) and the working
  precision. The series stopping rule k > x² + 2 is an exact rational test (an earlier version of the checker
  used a float test; the two agree on every input used).
- **Reporting only.** Floats in messages and in fields named `*_float`, `gauss_worst_upper`, `entry_max_lhs_minus_rhs`
  and `phi_upper` of the output JSON are for reporting only.
