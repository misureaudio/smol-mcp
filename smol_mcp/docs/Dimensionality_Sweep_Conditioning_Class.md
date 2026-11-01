# Dimensionality Sweep (2 → 20D) on the Conditioning Class

Date: 2026-10-31 · Project: smol_mcp (D:\Source\hermes-dir\smol_mcp)

## Purpose

The 2D sweeps (original 17 + new 15 functions) showed island DE winning
13/15 at a fixed 6000-eval budget. Whether that dominance survives
dimensionality is the open question: population methods under-sample high-D
spaces, SA's random walk is isotropic, and a GP surrogate needs ~O(10·n)
points to model n-dimensional structure. This sweep measures quality-per-
compute degradation with n on the conditioning class.

## Setup

Functions (standard n-dim forms, all known min = 0; matyas is 2D-only by
definition and stays in the 2D baseline):

| function | form | bounds | condition number |
|---|---|---|---|
| rosenbrock | Σᵢ [100(xᵢ₊₁ − xᵢ²)² + (1 − xᵢ)²] | [−5.12, 5.12]ⁿ | — (curved valley tube) |
| bent-cigar | x₁² + 100·Σᵢ₌₂ⁿ xᵢ² | [−2, 2]ⁿ | 100ⁿ⁻¹ (2 scales) |
| ellipsoid | Σᵢ 10^(6(i−1)/(n−1)) xᵢ² | [−1, 1]ⁿ | 10⁶ (n distinct scales) |

n ∈ {2, 5, 10, 20}. **Fixed evaluation budget across all n** (the question is
"same compute, higher n"):

| optimizer | budget |
|---|---|
| DE (island, dps=30) | 3 islands × 20 pop × 100 gen = 6060 evals |
| GA (dps=30) | 60 pop × 100 gen = 6000 evals |
| SA (dps=30, log schedule, T0=1.0) | 6000 iters, neighbor_scale = 5% of range (fixed) |
| BO (float64, UCB κ=2.5) | 195 + 5 = 200 evals (its design budget) |

Custom-code mode (n-dim `objective(x)` + bounds) on every tool. The
single-draw table below is confirmed by the 5-draw multi-start (see the
scaled-control section).

## Results (best value found; lower is better)

| function | n | DE (6k) | GA (6k) | SA (6k) | BO (200) | best |
|---|---|---|---|---|---|---|
| rosenbrock | 2 | **4.98e-14** | 4.02e-04 | 5.18e-04 | 3.44e-02 | DE |
| rosenbrock | 5 | **0.5246** | 2.749 | 4.819 | 151.8 | DE |
| rosenbrock | 10 | 41.28 | 65.97 | **12.38** | 622.7 | SA |
| rosenbrock | 20 | 553.0 | 402.5 | **100.8** | 4294.9 | SA |
| bent-cigar | 2 | **2.77e-18** | 4.88e-05 | 1.08e-05 | 8.99e-08 | DE |
| bent-cigar | 5 | **1.39e-04** | 7.95e-02 | 1.62e-02 | 4.21e-02 | DE |
| bent-cigar | 10 | 1.482 | 5.003 | 0.4362 | **0.2224** | BO |
| bent-cigar | 20 | 62.03 | 16.37 | 3.066 | **1.962** | BO |
| ellipsoid | 2 | **2.07e-16** | 4.90e-03 | 0.2550 | 9.03e-03 | DE |
| ellipsoid | 5 | **4.92e-04** | 1.732 | 10.27 | 61.53 | DE |
| ellipsoid | 10 | **1.741** | 25.45 | 124.3 | 3038 | DE |
| ellipsoid | 20 | **378.7** | 979.8 | 381.2 | 14777 | DE |

Wins by (function, n): DE 8, SA 2 (rosenbrock n=10, 20), BO 2 (bent-cigar
n=10, 20), GA 0.

## Findings

1. **The n=2 dominance ranking inverts by n=10–20.** DE wins everything at
   n ≤ 5; at n=10–20 the winners are SA (rosenbrock, 5.5× better than DE at
   n=20) and BO (bent-cigar, 31× better than DE at n=20). DE degrades the
   fastest in absolute terms (16–18 orders of magnitude n=2→20) but starts
   10–14 orders better, so the curves cross rather than DE losing early.

2. **Why DE degrades: population under-sampling.** 60 individuals in 20D
   cannot represent a 20D landscape. DE's mutation vector (xᵣ₁ + F(xᵣ₂ −
   xᵣ₃)) is a difference of two random 20D points: its component along the
   useful direction (e.g. the rosenbrock valley axis) is a ~1/√20 fraction of
   its norm, and most of the step energy jumps across valley walls. The
   island model's diversification does not help when each island's population
   is too small to sample the space.

3. **Why SA wins rosenbrock in high-D: isotropic walk + still cooling.**
   SA's per-coordinate Gaussian step (0.512) projects onto the valley axis
   with the same std regardless of n, and the log schedule at 6000 iterations
   is still warm (effective budget > iteration count, as seen in the 2D
   sweeps). Rosenbrock's valley is wide (cross-valley curvature 100–200), so
   a fixed step size is not a fatal floor — unlike on the narrow
   conditioning functions. Note SA's fixed single neighbor_scale for all 20
   coordinates is a handicap (optimal per-coordinate step shrinks ~1/√n), so
   the win is conservative.

4. **Why BO wins bent-cigar in high-D: effective dimensionality, not n.**
   Bent-cigar has only **2 scales** (weak x₁, uniform strong block of 19
   identical coordinates) — its effective dimensionality is ~2 (x₁ + the
   strong block's norm). BO's 200 points can learn that structure even at
   n=20, and its adaptive exploration (UCB: exploit where σ̂ shrank) polishes
   better than SA's fixed 0.2 step, which stalls at a step-size floor
   (SA 3.07 vs BO 1.96). The **ellipsoid** has **20 distinct scales**
   (10⁰…10⁶) — effective dimensionality 20 — and BO collapses (14777 at
   n=20). The GP's model capacity, not the ambient dimension, is the binding
   constraint.

5. **Ellipsoid is the hardest for population methods: spread of curvatures,
   not peak condition number.** DE does *better* on bent-cigar (62.0) than on
   the ellipsoid (378.7) at n=20 despite bent-cigar's astronomically larger
   condition number (10⁴⁰ vs 10⁶). The ellipsoid's 20 geometrically spread
   coefficients (10⁰…10⁶) force 20 distinct population-spread adaptations;
   bent-cigar's 19 identical strong coordinates adapt in parallel. DE's
   per-coordinate spread adaptation handles *diagonal* conditioning well —
   the number of distinct scales is what kills it.

6. **At n=20, nobody solves these at 6000 evals.** The best results are
   SA-rosenbrock 100.8, BO-bent-cigar 1.96, DE-ellipsoid 378.7 — the table
   measures "how badly do we fail", not "who solves it". A scaled budget
   (∝ n) or larger populations would shift values but the ranking
   *mechanisms* (under-sampling, isotropy, GP capacity) are budget-independent.

## Practical guidance (measured)

- n ≤ 5, conditioning/smooth: island DE, fixed 6k evals.
- n ≥ 10, wide-valley (rosenbrock-type): SA with log schedule (still cooling
  at 6k; give it more iterations).
- n ≥ 10, few-scale separable (bent-cigar-type) with expensive evaluations:
  BO is competitive even at 200 evals.
- n ≥ 10, many-scale diagonal (ellipsoid-type): DE with a scaled budget
  (pop ~20, many generations) solves it to machine precision at 120k evals
  (see below); at 6k, DE is the least-bad.

## Scaled-budget control (n=20, 120k evals) — "algorithm vs under-resourcing"

To separate algorithm from under-resourcing, the same 3 functions at n=20
were re-run with 20× the budget: DE 3×400 pop (20×n) × 100 gen, GA 400 pop
× 300 gen, SA 120k iters (BO kept at its 200-eval design budget — GP O(m³)
fit + 20D acquisition make 4k infeasible).

**First result: naively scaling DE (bigger population) made it worse.**

| function | DE 6k | DE 120k (pop 400) | GA 6k | GA 120k | SA 6k | SA 120k | BO 200 |
|---|---|---|---|---|---|---|---|
| rosenbrock | 552.97 | **24,349** (44× worse) | 402.51 | 74.67 | 100.83 | 27.61 | 4294.9 |
| bent-cigar | 62.03 | **520.5** (8× worse) | 16.37 | 8.04 | 3.066 | 1.794 | 1.962 |
| ellipsoid | 378.7 | **1476** (4× worse) | 979.8 | 102.9 | 381.2 | 169.5 | 14777 |

GA and SA exploited the extra compute (2.3–10× better); DE's per-generation
best-so-far (CSV-verified, monotonically non-increasing, no engine artifact)
sat at the initial value for 20 generations, then crawled: 66,970 → 24,349
(rosenbrock).

**Mechanism: DE's mutation step is F·(population spread) — the population
*is* the step-size controller.** Pop 400/island in 20D keeps the spread
large (400 points cannot collapse), so trial vectors are huge jumps that
narrow valleys reject almost wholesale — 100 generations only partially
contracts the spread. Pop 20/island collapses the spread fast, shrinks the
steps, and more generations polish.

**Cross-check — same 120k evals, small population, many generations
(3×pop 20 × 2000 gen) vs big population (3×pop 400 × 100 gen),
5 independent draws each:**

| function | DE pop 400 × 100 gen | DE pop 20 × 2000 gen (median [min..max]) |
|---|---|---|
| rosenbrock | 24,349 | 0.121 [0.053..3.99] (best draw 0.012) |
| bent-cigar | 520.5 | **2.4e-25 .. 1.7e-24** (all 5 draws at the dps=30 floor) |
| ellipsoid | 1,476 | **1.1e-25 .. 5.1e-24** (all 5 draws at the dps=30 floor) |

Pop-20 wins all 8 independent draws (3 single-draw + 5 multi-start). On
bent-cigar and ellipsoid, DE pop20×2000g **solves the 20D problem to the
30-digit precision floor**. (Correction: earlier "O(1–10)" readings of
these runs, and the "SA beats DE on bent-cigar" conclusion, were a display
artifact — a 20-character truncation of the printed value dropped the
"e-25" exponent; verified against the CSV trajectories.)

**Convergence curve, DE pop 20 × 3 islands, n=20 (CSV final values, full
precision):**

| function | 12k evals (200g) | 60k (1000g) | 120k (2000g) |
|---|---|---|---|
| rosenbrock | 34.12 | 8.19 | 0.12 [0.053..3.99], 5 draws (prior: 0.012, 4.0) |
| bent-cigar | 2.38 | 9.50e-12 | ~1e-25 (precision floor, 5/5) |
| ellipsoid | 12.55 | 3.57e-11 | ~1e-25 (precision floor, 5/5) |

Convergence is a clean exponential once the population locks onto the
quadratic basin (bent-cigar trajectory: gen0 1.4e3 → gen200 2.7 → gen500
3e-4 → gen1000 1.2e-10 → gen2000 6e-25). The weak x₁ direction
(coefficient 1) polishes too — no flat-direction stall. Rosenbrock remains
the hard case: even at 120k evals DE sits at 0.05–4 — the curved valley,
not the conditioning, is what limits DE.

**Multi-start confirmation of the dimensionality sweep (5 draws, fixed
6k budget; median [min..max] per optimizer):**

| function | n | DE | GA | SA | BO | winner (draws) |
|---|---|---|---|---|---|---|
| rosenbrock | 2 | 5.75e-15 [1.7e-15..2.6e-13] | 3.6e-04 [2.2e-05..0.02] | 1.2e-04 [2.5e-05..5.9e-04] | 0.0344 | DE (5/5) |
| rosenbrock | 5 | 0.717 [0.203..1.01] | 4.60 [2.1..4.94] | 1.04 [0.582..4.32] | 151.8 | DE (4/5) |
| rosenbrock | 10 | 27.8 [19.4..30.6] | 43.9 [16.9..83.3] | **11.1 [9.59..11.8]** | 622.7 | SA (5/5) |
| rosenbrock | 20 | 534 [448..1.1e3] | 267 [233..702] | **40.1 [36.8..48.8]** | 4295 | SA (5/5) |
| bent-cigar | 2 | 1.83e-18 [6.7e-19..1e-17] | 3.2e-05 [2.9e-06..1.1e-04] | 4.4e-05 [1.7e-06..9.8e-05] | 8.99e-08 | DE (5/5) |
| bent-cigar | 5 | 6.34e-05 [2.6e-05..1.7e-04] | 0.449 [0.088..1.42] | 0.0351 [0.0054..0.079] | 0.0421 | DE (5/5) |
| bent-cigar | 10 | 0.525 [0.356..1.53] | 4.11 [1.91..10.2] | 0.577 [0.526..1.24] | **0.222** | BO (5/5) |
| bent-cigar | 20 | 60.6 [47.8..76.4] | 40.1 [18.7..47] | 3.03 [2.13..4.29] | **1.962** | BO (5/5) |
| ellipsoid | 2 | 3.37e-17 [1.6e-17..1.1e-16] | 7.2e-03 [1.3e-03..0.067] | 8.0e-03 [1.3e-03..0.04] | 9.03e-03 | DE (5/5) |
| ellipsoid | 5 | 4.05e-04 [1.1e-04..1.1e-03] | 9.62 [0.436..15.4] | 3.05 [0.922..10.3] | 61.53 | DE (5/5) |
| ellipsoid | 10 | 2.32 [1.34..4.22] | 86.1 [76.2..215] | 48.7 [28.7..57.2] | 3038 | DE (5/5) |
| ellipsoid | 20 | 279.8 [198..386] | 680 [284..1.4e3] | 916 [342..1.0e3] | 1.48e4 | DE (4/5) |

The single-draw sweep's win counts (DE 8, SA 2, BO 2, GA 0) are reproduced
exactly; the only flips are near-ties inside the margin of error
(rosenbrock n=5: SA won 1 draw, 0.582 vs DE 1.01; ellipsoid n=20: GA won 1
draw, 284 vs DE 386). The n=10/20 inversions (SA-rosenbrock,
BO-bent-cigar) hold 5/5.

**Note on BO:** the BO tool's `random_state` was hardcoded to 1 (now the
default of its `seed` parameter), so its results are deterministic, not
draws — the identical BO values across "draws" above are the same run. Its
bent-cigar-20D value (1.962) is a reproducible fact under this
configuration, not a lucky sample. The `seed` parameter (added after the
sweeps) now enables true BO multi-start.

**Final n=20 leaderboard (120k-eval budget, 5 draws):**

| function | 1st | 2nd | 3rd |
|---|---|---|---|
| rosenbrock | **DE (pop20×2000g) 0.12** [0.053..3.99] | SA 28.8 [27.1..29.3] | GA 73.0 [50.3..100] |
| bent-cigar | **DE (pop20×2000g) ~1e-25** (solved, 5/5) | SA 1.59 [0.83..1.72] | BO 1.962 (deterministic) |
| ellipsoid | **DE (pop20×2000g) ~1e-25** (solved, 5/5) | SA 110 [98..178] | GA 206 [118..225] |

DE wins all three at 120k evals — solving bent-cigar and ellipsoid to the
30-digit floor and beating SA ~240× on rosenbrock.

**Library guidance (revised):** when scaling DE to n ≳ 10, scale
*generations*, not population — keep ~20–40 individuals/island and spend
the budget on generations (step shrinkage is the convergence mechanism).
A bigger population at fixed generations is actively harmful on
narrow-valley functions. At 120k evals, DE solves diagonal-conditioned 20D
problems (bent-cigar, ellipsoid) to machine precision; curved-valley
problems (rosenbrock) remain the limiting case. BO is a deterministic
single-shot tool under its current hardcoded seed.

## Artifacts

- Sweep: `C:\Users\MATTIA\AppData\Local\hermes\cache\scratch\smol_mcp_dim_sweep\`
  (`summary.json` + 48 CSVs), script `run_dim_sweep.py`.
- Scaled control: `C:\Users\MATTIA\AppData\Local\hermes\cache\scratch\smol_mcp_scaled_control\`
  (`summary.json`, `curve.json`, `sc_*` CSVs + `xcheck_de_*_pop20_g2000.csv`
  + `curve_de_*_pop20_{g200,g1000,g2000b}.csv`), script `run_scaled_control.py`.
- Multi-start (5 draws): `C:\Users\MATTIA\AppData\Local\hermes\cache\scratch\smol_mcp_multistart\`
  (`rows.json` + 300 CSVs), script `run_multistart.py` (resumable).
- 2D baselines: `...\smol_mcp_sweep\`, `...\smol_mcp_new_sweep\`,
  `docs/Four_Way_Sweep_New_Benchmark_Set.md`.

## Caveats

- The dimensionality sweep and scaled control are confirmed by 5 independent
  draws; the 2D baselines (31 functions) by 5–10 draws (see
  `Four_Way_Sweep_New_Benchmark_Set.md`). BO is deterministic under its
  default `seed=1` (reproducible, one sample per seed); the `seed` parameter
  (added after the sweeps) enables true BO multi-start — all BO values above
  are seed=1.
- Fixed 6000-eval budget at all n: by design, to isolate dimensionality —
  not a "fair per-parameter" budget.
- SA's single absolute neighbor_scale across all coordinates is a fixed
  protocol handicap in high-D.
- DE population (60 total) is below the usual 10–20×n guideline for n ≥ 10;
  part of DE's degradation is the fixed population, not the algorithm alone.
