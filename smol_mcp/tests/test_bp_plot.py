"""Tests for the bulletproof function plotter (nd_functions + BPPlotTool).

Run:  <smol_mcp>\\.venv\\Scripts\\python.exe tests\\test_bp_plot.py
Exit code 0 = all checks passed.
"""
import sys, os, math

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # smol_mcp/
sys.path.insert(0, BASE)
import numpy as np
import nd_functions as nd

TESTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_work")
os.makedirs(TESTDIR, exist_ok=True)

results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if not ok else ""))


def compile_lib_2d(name):
    """Compile the library's 2-D numpy code for `name` -> callable."""
    import tools_lib as tlib
    d = tlib.BULLETPROOF_FUNCTIONS[name]
    g, l = {"np": np, "math": math}, {}
    exec(d["code"].strip(), g, l)
    return l["objective"]


print("=" * 70)
print("SECTION 1: registry n=2 cross-validation vs library 2-D code")
print("=" * 70)

rng = np.random.default_rng(0)
for name, entry in nd.ND_REGISTRY.items():
    lib_fn = compile_lib_2d(name)
    lo, hi = entry["bounds"]
    worst = 0.0
    for _ in range(50):
        p = rng.uniform(lo, hi, size=2)
        f_nd = entry["fn"](p)
        f_lib = lib_fn(p)
        scale = max(1.0, abs(f_lib))
        worst = max(worst, abs(f_nd - f_lib) / scale)
    check(f"n=2 cross-validation: {name}", worst < 1e-9, f"worst rel diff {worst:.2e}")
    # minimizer sanity: f(min_x) == min_f
    mx = entry["min_x"](2)
    check(f"n=2 minimizer: {name}",
          abs(entry["fn"](mx) - entry["min_f"](2)) < 1e-8 * max(1.0, abs(entry["min_f"](2))))

print()
print("=" * 70)
print("SECTION 2: surface mode (2-D library functions)")
print("=" * 70)
from bp_plot_tool import BPPlotTool

tool = BPPlotTool(safe_dir=TESTDIR)

out = tool.forward(function="rosenbrock", output_filename="surf_rosenbrock.png")
p = os.path.join(TESTDIR, "surf_rosenbrock.png")
check("surface: rosenbrock PNG written",
      out.startswith("Plot saved") and os.path.exists(p) and os.path.getsize(p) > 20_000, out[:200])
check("surface: min on grid reported", "min on grid: 0" in out, out[:300])

out = tool.forward(function="ellipsoid", output_filename="surf_ellipsoid.png",
                   window=[-0.5, 0.5, -0.01, 0.01])
check("surface: windowed ellipsoid", out.startswith("Plot saved"), out[:200])

out = tool.forward(function="nope", output_filename="x.png")
check("surface: unknown function rejected", "unknown 2-D function" in out, out[:200])
out = tool.forward(function="rosenbrock", output_filename="x.jpg")
check("surface: bad extension rejected", "must end in .png" in out, out[:200])
out = tool.forward(function="rosenbrock", output_filename="x.png", mode="bogus")
check("surface: unknown mode rejected", "unknown mode" in out, out[:200])
out = tool.forward(function="ackley", output_filename=None)
check("surface: auto-generated filename",
      "bp_ackley_surface.png" in out and os.path.exists(os.path.join(TESTDIR, "bp_ackley_surface.png")), out[:200])

print()
print("=" * 70)
print("SECTION 3: slice mode (2-D assigned projections of n-D functions)")
print("=" * 70)

# ellipsoid n=5, free x0-x1, base zeros: slice = x0^2 + 1e6*x1^2 (x2..x4 pinned 0)
out = tool.forward(function="ellipsoid", output_filename="slice_ell5.png",
                   mode="slice", n=5, dims=[0, 1], grid=51)
check("slice: ellipsoid n=5 written",
      out.startswith("Plot saved") and os.path.getsize(os.path.join(TESTDIR, "slice_ell5.png")) > 20_000, out[:200])
check("slice: min at origin", "min on grid: 0" in out, out[:300])

# rosenbrock n=3, free x0-x2, base = minimizer [1,1,1]:
# slice(x0,x2) = 100*(1-x0^2)^2 + (1-x0)^2 + 100*(x2-1)^2 -> min 0 at (1,1)
out = tool.forward(function="rosenbrock", output_filename="slice_ros3.png",
                   mode="slice", n=3, dims=[0, 2], base_is_minimizer=True, grid=51)
check("slice: rosenbrock n=3 through minimizer",
      "min on grid: 0" in out and "base=[1.0, 1.0, 1.0]" in out, out[:400])

# value check: slice of bent-cigar n=4 at (x0=0.5, x3=0.25), base zeros
# = 0.25 + 100*0.0625 = 6.5
e = nd.ND_REGISTRY["bent-cigar"]
x = [0.0, 0.0, 0.0, 0.0]
x[0], x[3] = 0.5, 0.25
check("slice: value consistency (bent-cigar n=4)",
      abs(e["fn"](x) - (0.25 + 100 * 0.0625)) < 1e-12)

out = tool.forward(function="ellipsoid", output_filename="bad.png", mode="slice",
                   n=3, dims=[1, 1])
check("slice: duplicate dims rejected", "distinct indices" in out, out[:200])
out = tool.forward(function="ellipsoid", output_filename="bad.png", mode="slice",
                   n=3, dims=[0, 1], base_point=[0.0, 0.0])
check("slice: wrong-length base rejected", "must have 3 values" in out, out[:200])
out = tool.forward(function="shubert", output_filename="bad.png", mode="slice", n=4)
check("slice: n-D-ineligible function rejected", "no n-D generalization" in out, out[:200])

print()
print("=" * 70)
print("SECTION 4: path mode (parametric cross-sections)")
print("=" * 70)

# rosenbrock n=10: from zeros to the minimizer [1]*10 -> f drops to 0 at t=1
out = tool.forward(function="rosenbrock", output_filename="path_ros10.png",
                   mode="path", n=10, grid=201)
check("path: rosenbrock n=10 written",
      out.startswith("Plot saved") and os.path.getsize(os.path.join(TESTDIR, "path_ros10.png")) > 15_000, out[:200])
check("path: min at t=1.000 with f≈0",
      "min on path: 0" in out and "t=1.000" in out, out[:300])

# griewank n=6: explicit endpoints, min at t=0.5 (midpoint = zeros = minimizer)
out = tool.forward(function="griewank", output_filename="path_griew6.png",
                   mode="path", n=6,
                   path_from=[1.0] * 6, path_to=[-1.0] * 6, grid=201)
check("path: griewank n=6 symmetric, min at t=0.5",
      "t=0.500" in out and "min on path: 0" in out, out[:300])

out = tool.forward(function="ellipsoid", output_filename="bad.png", mode="path",
                   n=3, path_from=[0.0])
check("path: wrong-length endpoint rejected", "must each have 3 values" in out, out[:200])

print()
print("=" * 70)
print("SECTION 5: contour mode (standalone high-res contour)")
print("=" * 70)

out = tool.forward(function="rosenbrock", output_filename="contour_ros.png",
                   mode="contour", grid=101)
check("contour: rosenbrock written",
      out.startswith("Plot saved") and os.path.getsize(os.path.join(TESTDIR, "contour_ros.png")) > 15_000, out[:200])
check("contour: min on grid reported", "min on grid: 0" in out, out[:300])

out = tool.forward(function="ellipsoid", output_filename="contour_ell5.png",
                   mode="contour", n=5, dims=[0, 1], grid=51)
check("contour: n=5 slice written with base",
      out.startswith("Plot saved") and "base=[0.0, 0.0, 0.0, 0.0, 0.0]" in out, out[:300])

out = tool.forward(function="rastrigin", output_filename="contour_rast.png",
                   mode="contour", window=[-5, 5, -5, 5], grid=201)
check("contour: windowed rastrigin 201x201", out.startswith("Plot saved"), out[:200])

out = tool.forward(function="eggholder", output_filename="bad.png", mode="contour", n=4)
check("contour: n-D-ineligible at n>2 rejected", "no n-D generalization" in out, out[:200])
out = tool.forward(function="rosenbrock", output_filename="bad.png", mode="contour",
                   n=3, dims=[2, 2])
check("contour: duplicate dims rejected (n>2)", "distinct indices" in out, out[:200])

print()
passed = sum(1 for _, ok in results if ok)
print(f"TOTAL: {passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
