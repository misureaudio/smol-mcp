# Bulletproof Function Plotting

The `plot_bulletproof_function` MCP tool (registered in `mcp_server.py`,
implementation in `bp_plot_tool.py`, n-D registry in `nd_functions.py`)
produces PNG figures of the benchmark functions in three modes.

PNGs land in `C:\Users\MATTIA\smol_mcp_data\plots\` (override:
`SMOL_MCP_PLOTS_DIR`) and are delivered inline by the gateway.

## Modes

### 1. `surface` — native 2-D functions (all 31 library functions)

3-D surface + filled contour of the function over its full 2-D domain
(or a sub-rectangle via `window`).

```
plot_bulletproof_function(function="rosenbrock", output_filename="ros.png")
plot_bulletproof_function(function="eggholder", output_filename="egg.png",
                          window=[-10, 10, -10, 10])   # zoom: full range is ±512
```

### 2. `slice` — 2-D assigned projections of n-D functions

Pick an n>2 function, choose two free dimensions (`dims=[i, j]`); the
remaining n−2 coordinates are fixed at an assigned base point:

- `base_point=None` (default) → all zeros
- `base_point=[...]` → explicit n values
- `base_is_minimizer=true` → the known minimizer

```
# ellipsoid in 5-D, plane x0-x1, other dims pinned at 0:
plot_bulletproof_function(function="ellipsoid", mode="slice", n=5,
                          dims=[0, 1], output_filename="ell5.png")

# rosenbrock in 3-D, plane x0-x2, pinned at the minimizer [1,1,1]:
plot_bulletproof_function(function="rosenbrock", mode="slice", n=3,
                          dims=[0, 2], base_is_minimizer=true, output_filename="ros3.png")
```

Useful for seeing how a function looks along *any* pair of coordinates,
e.g. a slice through the point an optimizer returned.

### 3. `contour` — standalone high-resolution 2-D contour

Filled contours (50 levels) + 20 fine unlabeled lines + 3 labeled line
contours at the 25/50/75 percentiles of the value range (labeling more
levels collides on periodic landscapes like rastrigin; the colorbar
carries the full scale) + the minimum on grid marked with a red star. Best for seeing local minima,
ridges and the overall landscape structure that 3-D views hide.

- 2-D: any of the 31 library functions (default `n=2`)
- n>2: a slice of an n-D function (same `dims` / `base_point` /
  `base_is_minimizer` parameters as `slice`)

```
plot_bulletproof_function(function="rastrigin", mode="contour",
                          window=[-5, 5, -5, 5], grid=201,
                          output_filename="rast.png")

# 5-D rosenbrock, plane x0-x2 pinned at the minimizer:
plot_bulletproof_function(function="rosenbrock", mode="contour", n=5,
                          dims=[0, 2], base_is_minimizer=true,
                          output_filename="ros5.png")
```

### 4. `path` — parametrized 1-D cross-sections

f(t) = f(a + t·(b−a)), t ∈ [0,1], along a straight path through n-D space.
The curve minimum (t*, f*) is marked.

- `path_from` (default: zeros), `path_to` (default: known minimizer)

```
# origin -> global minimizer of rosenbrock in 10-D:
plot_bulletproof_function(function="rosenbrock", mode="path", n=10,
                          output_filename="ros10.png")

# explicit endpoints:
plot_bulletproof_function(function="griewank", mode="path", n=6,
                          path_from=[1,1,1,1,1,1], path_to=[-1,-1,-1,-1,-1,-1],
                          output_filename="griew6.png")
```

## n-D registry (slice / path modes)

Nine functions have standard n-D generalizations (numpy, float64). Every
entry is cross-validated at n=2 against the library's own 2-D code in
`tests/test_bp_plot.py` (50 random points per function, rel. tol. 1e-9):

| function | bounds | minimizer x* | f(x*) |
|---|---|---|---|
| rosenbrock | [−5.12, 5.12] | (1, 1, …, 1) | 0 |
| bent-cigar | [−2, 2] | 0 | 0 |
| ellipsoid | [−1, 1] | 0 | 0 |
| rastrigin | [−5.12, 5.12] | 0 | 0 |
| griewank | [−600, 600] | 0 | 0 |
| ackley | [−32.768, 32.768] | 0 | 0 |
| schwefel | [−500, 500] | (420.9687, …) | ≈ 2.5e−5 · n¹ |
| zakharov | [−5, 5] | 0 | 0 |
| styblinski-tang | [−5, 5] | (−2.903463, …) | −39.1661657 · n |

¹ The library's 2-D constant 837.9658 is a rounded value of
2 × 418.98288727…, so the true minimum of the (consistent) n-D form
418.9829·n − Σ xᵢ sin√|xᵢ| is slightly positive.

Zakharov uses the **library's coefficient convention** (s = Σ (i/2)·xᵢ,
i.e. 0.5, 1.0, 1.5, …), not the √i variant found in some literature —
the n=2 cross-validation pins it to the library.

The other 22 library functions (eggholder, shubert, holder table,
langermann, easom, himmelblau, six-hump camel, goldstein-price, schaffer
n. 2/4, gramacy & lee, bukin n. 6, penalty (kornkorn), cross-in-tray,
drop-wave, levy, lev n. 13, matyas, mogadish, plateau, dejong step, …)
have no canonical n-D version — they remain plottable in `surface` mode
only.

## Parameters

| param | type | default | meaning |
|---|---|---|---|
| `function` | string | — | library name (surface) or registry name (slice/path) |
| `output_filename` | string | auto | PNG name; auto = `bp_<function>_<mode>.png` |
| `mode` | string | `surface` | `surface` \| `slice` \| `contour` \| `path` |
| `n` | int | 2 | dimension (slice/path) |
| `dims` | [int, int] | [0, 1] | free dimensions (slice) |
| `base_point` | number[] | zeros | fixed coordinates (slice) |
| `base_is_minimizer` | bool | false | slice base = known minimizer |
| `path_from` | number[] | zeros | path start (path) |
| `path_to` | number[] | minimizer | path end (path) |
| `grid` | int | 101 | resolution, clamped to [11, 401] |
| `window` | [lo, hi, lo, hi] | full bounds | sub-rectangle for the 2 free dims |

## Notes

- **Precision:** float64, visualization grade by design — the `_mp`
  (mpmath) functions are deliberately not routed through the plotter.
- **Lazy matplotlib:** imported inside the call (Agg backend) so MCP
  server startup stays light.
- **Wide-range functions:** griewank (±600), eggholder (±512), schaffer
  n. 2/4 (±100), schwefel (±500) — the full-domain default is a shape
  overview; use `window` (or a slice/path around the minimizer) for
  detail.
- **Tests:** `tests/test_bp_plot.py` (42 checks: 18 registry
  cross-validation + 7 surface + 7 slice + 4 path + 6 contour), run with the
  smol-mcp venv python; exit code 0 = all pass.
