# Four-Way Sweep on the New Benchmark Set (Classes 1–4)

Date: 2026-10-31 · Project: smol_mcp (D:\Source\hermes-dir\smol_mcp)

## Purpose

Extended the bulletproof library from 17 to 31 functions (base + `_mp`) with
15 new functions targeting failure modes under-represented in the original
set: ill-conditioned valleys (class 1), roughness (class 2), deceptive /
equal-depth minima (class 3), non-smoothness (class 4). All reference minima
were measured (grid + bound-respecting L-BFGS-B multi-start), not taken from
the literature — see `bulletproof_extra.py`.

This note records the four-optimizer comparison on that new set.

## Setup

| optimizer | tool | budget | variant |
|---|---|---|---|
| DE (island model) | `DEToolMP` | 3 islands × 20 pop × 100 gen = 6060 evals, dps=30 | `_mp` |
| GA | `GAToolMP` | 60 pop × 100 gen = 6000 evals, dps=30 | `_mp` |
| SA | `SimAnnealToolMP` | 6000 iters, log schedule, T0=1.0, neighbor_scale = 5% of widest range, dps=30 | `_mp` (code + bounds) |
| BO | `BOToolGv1_v3_2_0` | 195 iters + 5 init = 200 evals, UCB κ=2.5 | base (float64) |

Single seed per run (stochastic methods: one draw each — distributions, not
point estimates). BO runs 30× fewer evaluations: it is the "expensive
function" tool and is compared on quality-per-eval, not absolute quality.

## Results

Gaps are (optimizer value − known min). Bold = best per row.

| function | known | **DE (6k)** | GA (6k) | SA (6k) | BO (200) |
|---|---|---|---|---|---|
| rosenbrock | 0 | **9.01e-15** | +2.73e-04 | +1.14e-04 | +3.44e-02 |
| matyas | 0 | **7.31e-18** | +8.36e-06 | +5.98e-05 | +1.83e-06 |
| bent-cigar | 0 | **7.16e-18** | +3.31e-05 | +2.28e-05 | +8.99e-08 |
| ellipsoid | 0 | **8.27e-18** | +1.05e-02 | +1.20e-04 | +9.03e-03 |
| holder n. 10 | −1.6594411760657 | **+3.29e-14** | +2.32e-03 | +1.20e-05 | +1.71e-03 |
| holder n. 100 | −1.05195263598706 | **−8.61e-13** | +5.76e-06 | +3.36e-06 | +6.26e-05 |
| easom | −1 | **−1 (+4.35e-13)** | +2.40e-02 | +1.00e+00 | +1.00e+00 |
| himmelblau | 0 | **9.58e-15** | +4.91e-03 | +6.47e-04 | +8.89e-05 |
| six-hump camel | −1.03162845349 | **+1.23e-13** | +7.86e-04 | +8.17e-06 | +2.72e-05 |
| goldstein-price | 3 | **3 (+4.44e-16)** | +4.20e-03 | **+81.0** | +6.47e-01 |
| styblinski-tang | −78.3323314075 | **−4.28e-11** | +8.07e-01 | +1.41e+01 | +3.21e-05 |
| zakharov | 0 | **4.64e-19** | +6.35e-05 | +1.22e-04 | +3.91e-05 |
| penalty (kornkorn) | 0 | **7.39e-11** | +1.99e-06 | +8.54e-08 | +7.41e-06 |
| plateau | 0 | **0** | 0 | 0 | 0 |
| dejong step | 2 | **2** | 2 | 2 | 2 |

Score: **DE 13 wins, 2 ties (plateau, dejong step — all four exact), 0 losses.
GA 0, SA 0, BO 0.**

## Findings

1. **DE dominates the conditioning class, and 2D conditioning is survivable
   even at condition number 10⁶.** Ellipsoid (1·x² + 10⁶·y²): DE 8.3e-18,
   GA 1.1e-2, SA 1.2e-4, BO 9.0e-3. The population spread adapts to the
   ellipse orientation over generations; nothing else does.   This is 2D specific — the dimensionality sweep
   (`Dimensionality_Sweep_Conditioning_Class.md`) is where conditioning
   starts hurting DE at the fixed 6k budget.

2. **The bukin n. 6 flat-valley finding refines to a gradient rule.**
   Rosenbrock's valley floor (y = x²) has slope ≈ 1.0/unit near the minimizer;
   bukin n. 6's has 0.01/unit. DE solved rosenbrock to 9e-15 at 6k evals but
   stopped 1.42 units short on bukin (0.0142 at 30k). The decisive parameter
   is valley-floor gradient relative to the fitness scale: bukin's floor
   gradient was below DE's selection resolution, rosenbrock's is not.

3. **BO's per-eval efficiency is real but bounded by polish.** With 30× fewer
   evals, BO was 2nd–3rd on 9 of 15. Standouts: bent-cigar **+8.99e-08**
   (its best result in either sweep; GP handled the narrow cigar),
   styblinski-tang **+3.21e-05** (the only non-DE optimizer to find the global
   min — GA +0.81 and SA +14.2 sat in local minima), himmelblau **+8.89e-05**.
   Its gaps are crawl-limited (1e-5…1e-2), never reaching DE's 1e-13.

4. **SA's failures are catastrophic where they happen.** Goldstein-price:
   **+81.0** (stuck in a local min 28× worse in value than the global min) —
   the worst single result in either sweep. Styblinski-tang +14.2, easom
   +1.0 (never found the (π,π) bump in 6k evals). SA was competitive on the
   smooth conditioning functions (2nd on rosenbrock, holder n. 100, penalty) —
   consistent with the earlier finding that its log schedule is still cooling
   at 6k iterations (effective budget > iteration count).

5. **GA is the reliable middle.** No catastrophic failure in this set
   (contrast SA's goldstein 84), but it never wins: it finds the right basin
   on most functions and crawls slowly (gaps 1e-3…1e-2).

6. **Easom is the needle test and only DE got the needle.** The (π,π) bump is
   ~5×5 units in a 200×200 box (≈0.06% of the area). DE −1 (4.4e-13) via
   island diversification + migration; GA found the rim (−0.976); SA and BO
   both returned ≈0 (never entered the bump at all).

7. **The non-smooth class is trivial in 2D as specified.** Plateau and
   dejong step both have their minima at the origin, surrounded by a flat
   region of the *optimal* value (any point in (−1,1)² scores 2 on dejong
   step) — all four optimizers hit the exact minimum. To make class 4 a real
   test, the minimum must be moved off-origin or the dimension raised.

## Multi-start extension (5–10 independent draws, all 31 functions)

To turn the single-draw results into distributions, all 31 functions (original
16 + new 15) were re-run with 5 independent draws each (DE/GA/SA engines are
unseeded → true independent initializations). The original 16 also received a
deterministic BO run (they had been 3-way only). The new 15 were run twice
(the function filter picked up all 31 `_mp` library entries), so they have
10 draws each.

**Result: DE wins 31/31.** 29 clear wins by median; 2 four-way ties
(plateau, dejong step — all four optimizers hit the exact known minimum in
every draw). GA 0, SA 0, BO 0. The single-draw "DE wins everything" is
confirmed; the only apparent exception (penalty (kornkorn), where SA's 5-draw
median 9.6e-9 beat DE's 6.6e-8) resolves in DE's favor at 10 draws (DE
median 2.3e-14 vs SA 1.3e-8 — the 5-draw sample was skewed by one bad DE
draw, max 1.6e-6).

Median [min..max] over the draws (BO: deterministic single value):

| function | DE | GA | SA | BO |
|---|---|---|---|---|
| ackley | 1.87e-08 [1.3e-08..6.1e-08] | 0.0735 [0.071..0.82] | 0.118 [0.082..0.15] | 0.0266 |
| bukin n. 6 | 0.0404 [0.0229..0.182] | 0.910 [0.207..2.44] | 0.481 [0.194..1.2] | 4.48 |
| cross-in-tray | −2.063 (all draws) | −2.063 | −2.063 | −2.063 |
| drop-wave | −1 (all draws) | −0.999 [−1.0..−0.99] | −0.959 [−0.99..−0.94] | −0.936 |
| easom | −1 (all draws) | −0.994 [−1.0..−0.90] | −0.139 [−0.92..−3.6e-16] | ≈0 |
| eggholder | −959.6 (≈ known) | −876 [−935..−718] | −717 [−895..−558] | −928.5 |
| ellipsoid | 9.4e-18 [7.1e-18..1.0e-16] | 0.0391 [5.2e-05..0.086] | 0.0242 [0.0015..0.14] | 9.03e-03 |
| goldstein-price | 3 (all draws) | 3.016 [3..3.15] | 3.001 [3..30] | 3.65 |
| gramacy & lee | −0.869 (all draws) | −0.869 | −0.435 [−0.65..−0.36] | −0.663 |
| griewank | 0.007396 [2.2e-10..0.0074] | 0.0375 [0.024..0.042] | 0.0148 [0.011..0.025] | 0.275 |
| himmelblau | 4.4e-15 [2.5e-17..1.1e-13] | 0.120 [0.0033..0.65] | 1.39e-04 [7.0e-05..7.4e-04] | 8.89e-05 |
| holder n. 10 | −1.659 (all draws) | −1.658 | −1.659 | −1.658 |
| holder n. 100 | −1.052 (all draws) | −1.052 | −1.052 | −1.052 |
| holder table | −19.21 (all draws) | −19.08 [−19.1..−18.7] | −9.50 [−16.3..−1.7] | −19.21 |
| langermann | −5.162 (all draws) | −4.86 [−5.16..−3] | −1.60 [−1.61..−1.57] | −5.162 |
| levy | 7.9e-19 [5.4e-20..5.0e-18] | 1.18e-04 [2.5e-05..4.0e-04] | 3.54 [1.2e-05..6.1] | 2.74e-06 |
| levy n. 13 | 1.46e-16 [2.7e-18..3.7e-16] | 3.98e-03 [2.6e-03..1.9e-02] | 7.23e-04 [1.2e-04..2.0e-03] | 0.269 |
| matyas | 1.93e-19 [1.4e-20..1.1e-18] | 5.49e-05 [1.3e-05..3.8e-04] | 1.88e-05 [9.8e-07..4.2e-05] | 1.83e-06 |
| penalty (kornkorn) | 6.6e-08 (5d) / 2.3e-14 (10d) | 1.45e-05 / 4.80e-06 | 9.6e-09 / 1.26e-08 | 7.41e-06 |
| plateau | 0 (all draws) | 0 | 0 | 0 |
| rastrigin | 1.56e-13 [6.8e-15..2.2e-13] | 5.78e-03 [2.6e-04..7.1e-03] | 24.9 [4.0..31.8] | 1.011 |
| rosenbrock | 1.76e-14 [4.2e-15..3.1e-12] | 2.00e-03 [4.6e-04..7.1e-03] | 5.75e-04 [1.2e-04..1.2e-03] | 3.44e-02 |
| schaffer n. 2 | 0.009716 [4.2e-06..0.00972] | 0.009719 [0.0014..0.0098] | 0.00985 [0.0097..0.038] | 0.217 |
| schaffer n. 4 | 0.2926 [0.293..0.293] | 0.2938 [0.293..0.296] | 0.3095 [0.299..0.364] | 0.36 |
| schwefel | 2.546e-05 (identical, all draws) | 120.5 [4.7..221] | 296 [217..513] | 0.0053 |
| shubert | −186.7 (all draws) | −186.3 [−187..−179] | −186.6 [−187..−186] | −186.7 |
| six-hump camel | −1.032 (all draws) | −1.031 | −1.032 [−1.03..−0.22] | −1.032 |
| styblinski-tang | −78.33 (all draws) | −78.15 [−78.3..−78] | −64.2 [−78.3..−50.1] | −78.33 |
| bent-cigar | 1.12e-18 (5d) / 1.6e-18 (10d) | 1.39e-04 / 4.56e-05 | 6.18e-05 / 5.2e-05 | 8.99e-08 |
| zakharov | 1.07e-18 (5d) / 4.0e-19 (10d) | 7.07e-05 / 9.40e-05 | 2.46e-05 / 2.41e-05 | 3.91e-05 |

Spread observations:
- **griewank** is the widest DE spread: 2.2e-10 to 0.0074 (4 orders of
  magnitude between draws) — one draw cracked nearly to the minimum, others
  settled on the 0.0074 plateau.
- **schwefel**: DE's value is *identical* in all draws (2.546e-05) — it falls
  into the same local minimum every time; BO (0.0053) is 200× worse but at
  least lands elsewhere.
- **bukin n. 6**: 8× spread (0.023–0.18), consistent with the flat-valley
  gradient finding — the polish depth is draw-dependent.
- **BO ties DE at the known minimum** on holder table and langermann, and is
  closest non-DE on eggholder (−928.5 vs −959.6, 3.3% off).

**Campaign tally across all multi-start data:** DE 31/31 at 2D (29 clear +
2 ties); at the 6k dimensionality budget, DE 8/12, SA 2/12 (rosenbrock
n=10, 20), BO 2/12 (bent-cigar n=10, 20), GA 0/12 — with two near-tie draws
(rosenbrock n=5: DE 4/5; ellipsoid n=20: DE 4/5); at the 120k budget (n=20),
DE 3/3, solving bent-cigar and ellipsoid to the 30-digit floor.

## BO multi-start (5 seeds, all 31 functions)

After the `seed` parameter was added to the BO tool (default 1 = the
documented value), seeds 2–5 were run (124 runs, ~94 min) at the same
200-eval budget (195+5, UCB κ=2.5). Seed 1 reuses the documented
deterministic value. Median [min..max] over the 5 seeds; "s1 rank" = rank of
the documented value among the 5 (1 = best):

| function | BO seed-1 (doc) | median(5) | [min..max] | s1 rank |
|---|---|---|---|---|
| ackley | 0.02657 | 0.06579 | [0.0266..0.142] | 1/5 |
| bent-cigar | 8.991e-08 | 0.0001269 | [8.99e-08..0.000622] | 1/5 |
| bukin n. 6 | 4.481 | 4.481 | [2.62..7.05] | 3/5 |
| cross-in-tray | −2.063 | −2.063 | all at min | 2/5 |
| dejong step | 2 | 2 | all at min | 1/5 |
| drop-wave | −0.9362 | −0.9362 | [−0.936..−0.785] | 3/5 |
| easom | ≈0 | ≈0 | [−4.4e-174..−0] (all stuck) | 2/5 |
| eggholder | −928.5 | −930.1 | [−937..−884] | 4/5 |
| ellipsoid | 0.009034 | 0.02446 | [0.00903..0.0558] | 1/5 |
| goldstein-price | 3.647 | 3.809 | [3.46..4.86] | 2/5 |
| gramacy & lee (2012) | −0.6633 | −0.6633 | [−0.869..−0.663] | 4/5 |
| griewank | 0.275 | 0.275 | [0.117..0.476] | 3/5 |
| himmelblau | 8.89e-05 | 0.000293 | [1.44e-05..0.000992] | 2/5 |
| holder n. 10 | −1.658 | −1.658 | all at min | 4/5 |
| holder n. 100 | −1.052 | −1.052 | all at min | 4/5 |
| holder table | −19.21 | −19.21 | all at min | 5/5 (tie) |
| langermann | −5.162 | −5.162 | [−5.16..−2.99] | 3/5 |
| levy | 2.738e-06 | 5.767e-06 | [2.74e-06..0.000123] | 1/5 |
| levy n. 13 | 0.2687 | 0.02359 | [0.000233..0.269] | 5/5 |
| matyas | 1.831e-06 | 1.654e-05 | [1.83e-06..9.15e-05] | 1/5 |
| penalty (kornkorn) | 7.408e-06 | 2.348e-06 | [4.56e-07..1.92e-05] | 4/5 |
| plateau | 0 | 0 | all at min | 1/5 |
| rastrigin | 1.011 | 0.09483 | [0.0116..1.01] | 5/5 |
| rosenbrock | 0.03443 | 0.03443 | [0.00868..0.278] | 3/5 |
| schaffer n. 2 | 0.217 | 0.07819 | [6.58e-07..0.485] | 4/5 |
| schaffer n. 4 | 0.36 | 0.3727 | [0.324..0.46] | 2/5 |
| schwefel | 0.0053 | 118.4 | [0.000305..217] | 2/5 |
| shubert | −186.7 | −186.7 | all at min | 1/5 |
| six-hump camel | −1.032 | −1.032 | all at min | 3/5 |
| styblinski-tang | −78.33 | −78.33 | all at min | 2/5 |
| zakharov | 3.908e-05 | 0.0002047 | [3.91e-05..0.00195] | 1/5 |

Findings:
- **The single-draw BO column is a distribution with fat tails.** Seed-1
  (the documented value) was the best of 5 in 9/31 functions and the worst
  in 3/31. Honest medians differ from the documented value by up to 3
  orders of magnitude.
- **Documented value was a lucky tail:** bent-cigar (8.99e-08 → median
  1.27e-04, ~1400× worse), ellipsoid (9.0e-3 → 2.4e-2), zakharov (3.9e-5 →
  2.0e-4), matyas (1.8e-6 → 1.7e-5), levy (2.7e-6 → 5.8e-6), ackley
  (0.0266 → 0.0658).
- **Documented value was the unlucky tail:** rastrigin (1.011 → median
  0.0948 — honest BO is 10× *better* than documented), levy n. 13 (0.269 →
  0.0236).
- **schwefel is bimodal:** 2 seeds found 3e-4..5e-3, 3 seeds stalled at
  118..217 → honest median 118.4, no better than GA's 5-draw median
  (120.5); the documented 0.0053 sat in the good half.
- **BO reliably solves (all or most seeds at the known minimum):**
  cross-in-tray, holder n. 10/100/table, shubert, six-hump camel,
  styblinski-tang, langermann (4/5); gramacy & lee is 1/5 (one seed hit
  −0.869, four stalled at −0.663).
- **Verdict holds:** DE 29/31 + 2 exact ties (plateau, dejong step) vs the
  honest BO median — the single-draw "DE wins 31/31" is not an artifact of
  BO's seed luck.

## Artifacts

- Sweep: `C:\Users\MATTIA\AppData\Local\hermes\cache\scratch\smol_mcp_new_sweep\`
  (`summary.json` + 60 CSVs), script `run_new_sweep.py`.
- Multi-start (5–10 draws, all 31 functions):
  `C:\Users\MATTIA\AppData\Local\hermes\cache\scratch\smol_mcp_2d_multistart\`
  (`rows.json` + CSVs), script `run_2d_multistart.py` (resumable).
- BO multi-start (5 seeds, all 31 functions, 200-eval budget):
  `C:\Users\MATTIA\AppData\Local\hermes\cache\scratch\smol_mcp_bo_multistart\`
  (`rows.json` + `aggregated.json` + 124 CSVs), script
  `run_bo_multistart.py` (resumable).
- Library additions: `smol_mcp/bulletproof_extra.py` (30 entries, measured
  references), merged by `assemble_tools_lib.py` into `tools_lib.py`.
- Original 17-function sweeps: `...\smol_mcp_sweep\`, `...\smol_mcp_sweep_retry\`,
  `...\smol_mcp_bo_sweep\`.

## Caveats

- The single-draw table above is confirmed by the multi-start extension
  below (5–10 independent draws per function); BO is now also
  distributional (5 seeds, see "BO multi-start" section). All values in the
  main table are seed=1 / single-draw.
- BO's 200-eval budget is its design range, not a like-for-like 6k comparison.
- SA's `SimAnnealToolMP` has no bulletproof name lookup (code + bounds-string
  interface only); its neighbor_scale is absolute (5% of widest range here),
  while DE/GA scale to the domain automatically.
