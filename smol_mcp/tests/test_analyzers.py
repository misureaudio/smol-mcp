"""Tests for the smol-mcp analyzer tools (SA/GA/BO).

Level 1: synthetic CSVs that force each diagnostic branch.
Level 2: real optimizer runs -> analyzers (incl. MP-format CSV support).

Run:  <smol_mcp>\\.venv\\Scripts\\python.exe tests\\test_analyzers.py
Exit code 0 = all checks passed.
"""
import sys, os, csv, math

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # smol_mcp/
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "vendor"))
import tools_lib as tlib
import tools_lib_mp as t

# Replicate mcp_server._load_engines(): the numpy SA/GA tools reference their
# engines as module-level globals that the MCP server binds at startup.
from core import SimulatedAnnealing
import neighbors as sa_neighbors
tlib.SimulatedAnnealing = SimulatedAnnealing
tlib.sa_neighbors = sa_neighbors
from genetic import RealGeneticAlgorithm
tlib.RealGeneticAlgorithm = RealGeneticAlgorithm
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import (UpperConfidenceBound, ExpectedImprovement,
                                   ProbabilityOfImprovement)
tlib.BayesianOptimization = BayesianOptimization
tlib.UpperConfidenceBound = UpperConfidenceBound
tlib.ExpectedImprovement = ExpectedImprovement
tlib.ProbabilityOfImprovement = ProbabilityOfImprovement

TESTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_work")
os.makedirs(TESTDIR, exist_ok=True)

sa_a = tlib.SAAnalysisTool(safe_dir=TESTDIR)
ga_a = tlib.GAAnalysisTool(safe_dir=TESTDIR)
bo_a = tlib.BOAnalysisTool(safe_dir=TESTDIR)

results = []


def check(name, out, *markers, absent=()):
    ok = all(m in out for m in markers) and all(m not in out for m in absent)
    results.append((name, ok, out if not ok else None))
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print("      expected:", markers, "absent:", absent)
        print("      got:", (out or "")[:600].replace("\n", " | "))


def wcsv(fname, header, rows):
    with open(os.path.join(TESTDIR, fname), "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)


# ============================ LEVEL 1: synthetic ============================
print("=" * 70)
print("LEVEL 1: synthetic CSVs, one per diagnostic branch")
print("=" * 70)

# ---- SA cases (header: iteration, energy, best_so_far, accepted, temperature, neighbor_scale, x0)
SA_H = ["iteration", "energy", "best_so_far", "accepted", "temperature", "neighbor_scale", "x0"]


def sa_rows(n, best_at, accept, t0=1.0, cool=0.999, scale=0.1, scale_varies=False):
    rows, best = [], 100.0
    for i in range(n):
        e = max(1.0 / (1 + i / 50.0), 0.01) if i < best_at else best
        best = min(best, e)
        acc = 1 if (i * 7919) % 100 < accept else 0
        sc = scale if not scale_varies else scale * math.exp(-i / (n / 3))
        rows.append([i, f"{e:.6f}", f"{best:.6f}", acc,
                     f"{t0 * cool ** i:.6f}", f"{sc:.6f}", "0.5"])
    return rows


wcsv("sa_quench.csv", SA_H, sa_rows(2000, best_at=60, accept=50))
wcsv("sa_slow.csv", SA_H, sa_rows(2000, best_at=1950, accept=50))
wcsv("sa_lowacc.csv", SA_H, sa_rows(2000, best_at=1000, accept=5))
wcsv("sa_highacc.csv", SA_H, sa_rows(2000, best_at=1000, accept=85))
wcsv("sa_stag.csv", SA_H, sa_rows(2000, best_at=500, accept=50))
wcsv("sa_adaptive.csv", SA_H, sa_rows(2000, best_at=1000, accept=50, scale_varies=True))
wcsv("sa_short.csv", SA_H, sa_rows(30, best_at=15, accept=50))
# MP SA CSV format: no best_so_far / accepted / neighbor_scale columns
wcsv("sa_mpfmt.csv", ["iteration", "energy", "temperature", "x0"],
     [[i, f"{100 / (1 + i)}", f"{1.0 * 0.999 ** i}", "0.1"] for i in range(500)])

check("SA quenching (best @3%)", sa_a.forward("sa_quench.csv"), "QUENCHING")
check("SA slow convergence (best @97.5%)", sa_a.forward("sa_slow.csv"),
      "SLOW CONVERGENCE", absent=["QUENCHING"])
check("SA low acceptance (<15%)", sa_a.forward("sa_lowacc.csv"), "Too low!")
check("SA high acceptance (>70%)", sa_a.forward("sa_highacc.csv"), "Too high!")
check("SA stagnation + restart rec", sa_a.forward("sa_stag.csv"),
      "stagnation period", "restart_stagnation")
# healthy late acceptance (50%) + varying scale -> the (now reachable)
# "worked perfectly" branch
check("SA adaptive scaling works (late-accept 50%)", sa_a.forward("sa_adaptive.csv"),
      "Adaptive scaling worked perfectly!")
check("SA <50 rows guard", sa_a.forward("sa_short.csv"), "Not enough data")
check("SA missing file", sa_a.forward("nope.csv"), "File not found")
check("SA bad extension", sa_a.forward("sa_quench.txt"), "must end with .csv")
# MP-format CSV: analyzer now derives the running best from energy
# (energy=100/(1+i) strictly decreases -> best at the end -> SLOW CONVERGENCE)
check("SA MP-format CSV now parses (derived best)", sa_a.forward("sa_mpfmt.csv"),
      "SIMULATED ANNEALING ANALYSIS", "Not logged by this engine", "SLOW CONVERGENCE",
      absent=["Analysis failed"])

# ---- GA cases (header: generation, best_energy, avg_energy, diversity, x0)
GA_H = ["generation", "best_energy", "avg_energy", "diversity", "x0"]


def ga_rows(n, collapse_at, pressure="normal"):
    rows = []
    for g in range(n):
        b = 10.0 * math.exp(-g / 30.0)
        d = max(1.0 - g / collapse_at, 0.001) if g < collapse_at else 0.001
        if pressure == "tight":
            a = b * 1.01
        else:
            a = b * (1 + 5 * d)
        rows.append([g, f"{b:.6f}", f"{a:.6f}", f"{d:.6f}", "0.0"])
    return rows


wcsv("ga_premature.csv", GA_H, ga_rows(100, collapse_at=10, pressure="tight"))
wcsv("ga_healthy.csv", GA_H, ga_rows(100, collapse_at=60))
wcsv("ga_notconv.csv", GA_H, [[g, f"{10 * math.exp(-g / 200):.6f}",
                               f"{20 * math.exp(-g / 200):.6f}", "0.5", "0.0"]
                              for g in range(100)])
wcsv("ga_tight.csv", GA_H, ga_rows(100, collapse_at=60, pressure="tight"))

check("GA premature convergence", ga_a.forward("ga_premature.csv"),
      "PREMATURE CONVERGENCE")
check("GA healthy convergence", ga_a.forward("ga_healthy.csv"),
      "CONVERGED", absent=["PREMATURE", "NOT CONVERGED"])
check("GA not converged", ga_a.forward("ga_notconv.csv"), "NOT CONVERGED")
check("GA high selective pressure", ga_a.forward("ga_tight.csv"),
      "HIGH SELECTIVE PRESSURE")
check("GA high variance", ga_a.forward("ga_healthy.csv"), "HIGH VARIANCE",
      absent=["HIGH SELECTIVE PRESSURE"])

# ---- BO cases (header: iteration, energy, best_so_far, x0, x1)
BO_H = ["iteration", "energy", "best_so_far", "x0", "x1"]


def bo_rows(n, spread, stalled):
    rows, best = [], 100.0
    for i in range(n):
        x0 = (i % 13) / 13.0 * 2 * spread - spread
        x1 = (i % 7) / 7.0 * 2 * spread - spread
        if stalled and i >= n // 2:
            e = best  # flat second half
        else:
            e = 100.0 * math.exp(-i / max(1, n / 4))
        best = min(best, e)
        rows.append([i, f"{e:.6f}", f"{best:.6f}", f"{x0:.4f}", f"{x1:.4f}"])
    return rows


wcsv("bo_lowexp.csv", BO_H, bo_rows(50, spread=0.1, stalled=False))
wcsv("bo_goodexp.csv", BO_H, bo_rows(50, spread=5.0, stalled=False))
wcsv("bo_stalled.csv", BO_H, bo_rows(50, spread=5.0, stalled=True))
wcsv("bo_learning.csv", BO_H, bo_rows(50, spread=5.0, stalled=False))
wcsv("bo_short.csv", BO_H, bo_rows(8, spread=5.0, stalled=True))

check("BO low exploration", bo_a.forward("bo_lowexp.csv"), "LOW EXPLORATION")
check("BO good exploration", bo_a.forward("bo_goodexp.csv"), "GOOD EXPLORATION")
check("BO stalled (flat 2nd half)", bo_a.forward("bo_stalled.csv"), "STALLED")
check("BO learning (improving)", bo_a.forward("bo_learning.csv"), "LEARNING",
      absent=["STALLED"])
# n<=10: honest "skipped" message, no STALLED and no false LEARNING
check("BO short-run gate (n<=10)", bo_a.forward("bo_short.csv"),
      "Convergence check skipped", absent=["STALLED", "LEARNING"])

# ============================ LEVEL 2: real runs ============================
print()
print("=" * 70)
print("LEVEL 2: real optimizer runs -> analyzers")
print("=" * 70)

# SA real: rosenbrock, fast exponential cooling
out = tlib.SimAnnealTool(safe_dir=TESTDIR).forward(
    function_code="rosenbrock", bounds=None, output_filename="real_sa.csv",
    max_iterations=3000, initial_temperature=10.0, cooling_rate=0.998,
    schedule_type="exponential", adapt_neighbor=False)
rep = sa_a.forward("real_sa.csv")
print("SA real run ->", (out.splitlines() or ["?"])[0])
print("analyzer verdict lines:", [l for l in rep.splitlines()
      if any(k in l for k in ("QUENCHING", "SLOW", "stagnation", "acceptance", "Healthy"))][:4])
check("SA real CSV parses + reports", rep, "SIMULATED ANNEALING ANALYSIS")

# GA real: ellipsoid 2D, 60 pop x 100 gen
out = tlib.GATool(safe_dir=TESTDIR).forward(
    function_code="ellipsoid", bounds=None, output_filename="real_ga.csv",
    pop_size=60, generations=100)
rep = ga_a.forward("real_ga.csv")
print("GA real run ->", (out.splitlines() or ["?"])[1] if len(out.splitlines()) > 1 else out[:120])
print("analyzer verdict lines:", [l for l in rep.splitlines()
      if any(k in l for k in ("CONVERGED", "PREMATURE", "PRESSURE", "VARIANCE", "Generations"))][:4])
check("GA real CSV parses + reports", rep, "GENETIC ALGORITHM REPORT")

# BO real: schwefel with seed=2 — the 5-seed campaign verified this seed
# stalls at ~217.15 (200-eval budget), i.e. a real "converged to a bad basin"
# run. (Deterministic: same value every time.)
out = tlib.BOToolGv1_v3_2_0(safe_dir=TESTDIR).forward(
    function_code="schwefel", bounds=None, output_filename="real_bo.csv",
    n_iter=195, init_points=5, seed=2)
rep = bo_a.forward("real_bo.csv")
print("BO real run (schwefel, seed=2) ->", (out.splitlines() or ["?"])[1]
      if len(out.splitlines()) > 1 else out[:120])
print("analyzer verdict lines:", [l for l in rep.splitlines()
      if any(k in l for k in ("EXPLORATION", "STALLED", "LEARNING", "Best Energy"))][:4])
check("BO real CSV parses + reports", rep, "BAYESIAN OPTIMIZATION REPORT")

# MP SA real: genuine MP CSV (no best_so_far/accepted) -> analyzer must now
# produce a full report with the derived running best.
# NOTE: SimAnnealToolMP has no bulletproof name lookup — it takes the
# function code + bounds string directly.
_mp_d = tlib.BULLETPROOF_FUNCTIONS["rosenbrock_mp"]
out = t.SimAnnealToolMP(safe_dir=TESTDIR).forward(
    function_code=_mp_d["code"], bounds=_mp_d["bounds"],
    output_filename="real_sa_mp.csv",
    schedule_type="log", dps=30, max_iterations=1000,
    initial_temperature="1.0", neighbor_scale="0.2")
rep = sa_a.forward("real_sa_mp.csv")
print("MP SA real run ->", (out.splitlines() or ["?"])[0])
print("analyzer verdict lines:", [l for l in rep.splitlines()
      if any(k in l for k in ("SIMULATED", "Not logged", "CONVERGENCE", "QUENCHING", "SLOW"))][:4])
check("MP SA real CSV now parses (gap closed)", rep,
      "SIMULATED ANNEALING ANALYSIS", "Not logged by this engine",
      absent=["Analysis failed"])

# ============================ summary ============================
n_pass = sum(1 for _, ok, _ in results if ok)
print()
print("=" * 70)
print(f"SUMMARY: {n_pass}/{len(results)} checks passed")
for name, ok, out in results:
    if not ok:
        print("  FAILED:", name)
print("=" * 70)
sys.exit(0 if n_pass == len(results) else 1)
