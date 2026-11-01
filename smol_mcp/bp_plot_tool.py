"""plot_bulletproof_function: surface / slice / contour / path plots.

- surface: 3-D surface + filled contour of a 2-D function (all 31 library 2-D functions).
- slice:   2-D assigned projection of an n-D function (nd_functions):
           dims=[i,j] free, all other coordinates fixed at base_point.
- contour: standalone high-resolution 2-D contour (filled + line contours,
           minimum marked); 2-D library functions, or a slice of an n-D
           function when n > 2.
- path:    1-D parametric cross-section f(t) = f(a + t*(b-a)), t in [0,1].

PNG output to the plots safe_dir. matplotlib is imported lazily (Agg)
to keep MCP server startup light.
"""
import os
import math
import numpy as np
from smolagents.tools import Tool

import nd_functions as nd
import tools_lib as tlib


def _compile_2d(name):
    d = tlib.BULLETPROOF_FUNCTIONS[name]
    g, l = {"np": np, "math": math}, {}
    exec(d["code"].strip(), g, l)
    return l["objective"], d["bounds"], d.get("global_min")


def _lazy_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


class BPPlotTool(Tool):
    name = "plot_bulletproof_function"
    description = """
    Plots bulletproof benchmark functions and saves a PNG.
    Modes:
    - surface: 3D surface + filled contour of a 2-D function (all 31 library 2-D functions).
    - slice:   2-D assigned projection of an n-D function: 2 free dims (dims=[i,j]),
               the remaining n-2 coordinates fixed at base_point (zeros by default,
               or the known minimizer with base_is_minimizer=true).
    - contour: standalone high-resolution 2-D contour (filled + labeled line
               contours, minimum on grid marked); any 2-D library function, or a
               slice of an n-D function when n > 2. Best for seeing local minima
               and ridges that 3-D views hide.
    - path:    1-D parametric cross-section f(t) = f(a + t*(b-a)), t in [0,1],
               from path_from (default zeros) to path_to (default known minimizer).
    n-D modes support: rosenbrock, bent-cigar, ellipsoid, rastrigin, griewank,
    ackley, schwefel, zakharov, styblinski-tang.
    Wide-range functions (griewank, eggholder, schaffer n. 2/4, schwefel):
    use `window` to zoom a sub-rectangle, e.g. window=[-2, 2, -2, 2].
    If output_filename is omitted it is auto-generated (bp_<function>_<mode>.png).
    """
    inputs = {
        "function": {"type": "string",
                     "description": "Function name, e.g. 'rosenbrock', 'ellipsoid'."},
        "output_filename": {"type": "string", "nullable": True,
                            "description": "PNG filename (must end in .png; auto-generated if omitted)."},
        "mode": {"type": "string", "nullable": True,
                 "description": "'surface' (default), 'slice', 'contour', or 'path'."},
        "n": {"type": "integer", "nullable": True,
              "description": "Dimension for slice/path/contour. Default 2 (slice at n=2 == surface)."},
        "dims": {"type": "array", "nullable": True,
                 "description": "Two free dimension indices for slice/contour(n>2), e.g. [0, 2]. Default [0, 1]."},
        "base_point": {"type": "array", "nullable": True,
                       "description": "n values fixing non-free coordinates (slice). Default: zeros."},
        "base_is_minimizer": {"type": "boolean", "nullable": True,
                              "description": "Slice: use the known minimizer as base point. Default false."},
        "path_from": {"type": "array", "nullable": True,
                      "description": "Path start (n values). Default: zeros."},
        "path_to": {"type": "array", "nullable": True,
                    "description": "Path end (n values). Default: known minimizer."},
        "grid": {"type": "integer", "nullable": True,
                 "description": "Grid resolution (surface/slice/contour: per axis; path: samples). Default 101."},
        "window": {"type": "array", "nullable": True,
                   "description": "[lo1, hi1, lo2, hi2] sub-rectangle for the 2 free dims. Default: full bounds."},
        "levels": {"type": "integer", "nullable": True,
                   "description": "Contour level count (surface/contour). Raise it in tight zooms to resolve fine texture below the default level spacing. Default: 40 (surface), 50 (contour)."},
    }
    output_type = "string"

    def __init__(self, safe_dir):
        super().__init__()
        self.safe_dir = safe_dir

    # ---------- shared validation ----------
    def _get_entry(self, function, mode, n):
        name = function.strip().lower()
        if mode in ("slice", "path") or (mode == "contour" and n > 2):
            if name not in nd.ND_REGISTRY:
                avail = ", ".join(sorted(nd.ND_REGISTRY))
                return None, f"Error: '{function}' has no n-D generalization. Available: {avail}."
            return nd.ND_REGISTRY[name], None
        if name not in tlib.BULLETPROOF_FUNCTIONS:
            return None, f"Error: unknown 2-D function '{function}'."
        fn, bounds, gmin = _compile_2d(name)
        return {"fn": fn,
                "bounds": [(b[0], b[1]) for b in bounds],
                "min_x": None, "min_f": gmin}, None

    # ---------- grid evaluation (shared by surface / slice / contour) ----------
    def _build_grid(self, entry, mode, n, dims, base_point, base_is_minimizer, grid, window):
        """Evaluate the function on the (grid x grid) plane of the 2 free dims.
        Returns (X, Y, Z, xlabel, ylabel, base_note, error) — error is None on success."""
        if mode == "slice" or (mode == "contour" and n > 2):
            if n < 2:
                return None, None, None, None, None, None, "Error: slice requires n >= 2."
            if len(dims) != 2 or dims[0] == dims[1] or not all(0 <= d < n for d in dims):
                return None, None, None, None, None, None, f"Error: dims must be two distinct indices in [0, {n - 1}], got {dims}."
            if base_is_minimizer:
                base = entry["min_x"](n)
            elif base_point is None:
                base = [0.0] * n
            else:
                base = [float(v) for v in base_point]
                if len(base) != n:
                    return None, None, None, None, None, None, f"Error: base_point must have {n} values, got {len(base)}."
            (lo, hi) = entry["bounds"]
            xilo, xihi, xjlo, xjhi = (window if window else (lo, hi, lo, hi))
            gi = np.linspace(xilo, xihi, grid)
            gj = np.linspace(xjlo, xjhi, grid)
            GI, GJ = np.meshgrid(gi, gj)
            Z = np.empty_like(GI)
            for r in range(grid):
                for c in range(grid):
                    x = list(base)
                    x[dims[0]] = GI[r, c]
                    x[dims[1]] = GJ[r, c]
                    Z[r, c] = entry["fn"](x)
            i, j = dims
            return GI, GJ, Z, f"x{i}", f"x{j}", f"base={np.array(base).round(3).tolist()}", None
        # 2-D library function (surface / contour at n=2)
        (lo, hi), (lo2, hi2) = entry["bounds"]
        x0lo, x0hi, x1lo, x1hi = window if window else (lo, hi, lo2, hi2)
        g = np.linspace(x0lo, x0hi, grid)
        h = np.linspace(x1lo, x1hi, grid)
        X, Y = np.meshgrid(g, h)
        # Library 2-D signature: objective(x) with x = [x, y]
        Z = np.vectorize(lambda a, b: entry["fn"]([a, b]))(X, Y)
        return X, Y, Z, "x", "y", None, None

    @staticmethod
    def _min_detail(X, Y, Z, grid, xlabel, ylabel, base_note=None):
        r, c = divmod(int(Z.argmin()), grid)
        detail = (f"min on grid: {Z.min():.6g} at {xlabel}={X[r, c]:.4g}, "
                  f"{ylabel}={Y[r, c]:.4g}")
        if base_note:
            detail += f"; {base_note}"
        return detail

    # ---------- 2-panel figure (surface / slice) ----------
    def _two_panel(self, name, entry, kind, n, dims, grid, out_png,
                   X, Y, Z, xlabel, ylabel, base_note, levels=None):
        plt = _lazy_plt()
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor=None, linewidth=0)
        if kind == "slice":
            i, j = dims
            ax.set_xlabel(f"x{i}")
            ax.set_ylabel(f"x{j}")
            ax.set_title(f"{name} (n={n}) — slice x{i}-x{j}")
        else:
            ax.set_title(f"{name} — surface")
        ax2 = fig.add_subplot(1, 2, 2)
        cs = ax2.contourf(X, Y, Z, levels=max(8, int(levels or 40)), cmap="viridis")
        fig.colorbar(cs, ax=ax2, label="f(x, y)")
        if kind == "slice":
            ax2.set_title(f"contour — {base_note}" if base_note else f"{name} — contour")
            fig.suptitle(f"{name} (n={n}) — 2-D assigned projection, free dims {dims}", y=1.02)
        else:
            ax2.set_title(f"{name} — contour")
            if entry.get("min_f") is not None:
                fig.suptitle(f"{name}  (known min: {entry['min_f']})", y=1.02)
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return self._min_detail(X, Y, Z, grid, xlabel, ylabel, base_note)

    # ---------- contour (standalone high-resolution 2-D contour) ----------
    def _contour(self, name, entry, kind, n, dims, grid, out_png,
                 X, Y, Z, xlabel, ylabel, base_note, levels=None):
        plt = _lazy_plt()
        fig, ax = plt.subplots(figsize=(8, 7))
        nlev = max(8, int(levels or 50))
        cs = ax.contourf(X, Y, Z, levels=nlev, cmap="viridis")
        # Fine unlabeled lines for detail + a sparse labeled set for readability
        ax.contour(X, Y, Z, levels=max(4, nlev * 20 // 50),
                   colors="k", linewidths=0.4 if nlev <= 100 else 0.25, alpha=0.5)
        # Label only the 25/50/75-percentile levels: on periodic landscapes
        # (rastrigin: 121 local minima) more labeled levels mean hundreds of
        # colliding inline labels; the colorbar carries the full value scale.
        zmin, zmax = float(Z.min()), float(Z.max())
        label_levels = ([zmin + f * (zmax - zmin) for f in (0.25, 0.5, 0.75)]
                        if zmax > zmin else [zmin])
        ct = ax.contour(X, Y, Z, levels=label_levels, colors="k",
                        linewidths=0.9, alpha=0.85)
        # Adaptive label precision: on tight zooms the value span can be far
        # below the 4th significant digit, so %.4g would print duplicate
        # labels (e.g. 1.2794/1.2798/1.2803 all -> "1.28"). Use as many
        # decimals as needed to distinguish the three labeled levels.
        if zmax > zmin:
            nd = max(1, int(math.ceil(-math.log10((zmax - zmin) / 3.0))))
            fmt = f"%.{nd}f"
        else:
            fmt = "%.4g"
        ax.clabel(ct, fmt=fmt, fontsize=9)
        fig.colorbar(cs, ax=ax, label="f")
        r, c = divmod(int(Z.argmin()), grid)
        ax.plot([X[r, c]], [Y[r, c]], "r*", ms=14, zorder=5,
                label=f"min on grid: {Z.min():.4g}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        title = f"{name} — contour"
        if kind == "slice":
            i, j = dims
            title += f" (n={n}, slice x{i}-x{j})"
        if base_note:
            title += f" — {base_note}"
        if entry.get("min_f") is not None:
            title += f"  (known min: {entry['min_f']})"
        ax.set_title(title)
        ax.legend(loc="upper right")
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return self._min_detail(X, Y, Z, grid, xlabel, ylabel, base_note)

    # ---------- path (parametric 1-D cross-section) ----------
    def _path(self, name, entry, n, path_from, path_to, grid, out_png):
        plt = _lazy_plt()
        if n < 2:
            return None, "Error: path requires n >= 2."
        a = [float(v) for v in path_from] if path_from else [0.0] * n
        b = [float(v) for v in path_to] if path_to else entry["min_x"](n)
        if len(a) != n or len(b) != n:
            return None, f"Error: path_from/path_to must each have {n} values."
        t = np.linspace(0.0, 1.0, grid)
        Y = np.array([entry["fn"]([a[k] + (b[k] - a[k]) * ti for k in range(n)]) for ti in t])
        k = int(Y.argmin())
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t, Y, "-", color="tab:blue", lw=1.5, label="f(t)")
        ax.plot([t[k]], [Y[k]], "r*", ms=14, label=f"min on path: {Y[k]:.4g} at t={t[k]:.3f}")
        ax.plot([0, 1], [Y[0], Y[-1]], "ko", ms=5)
        ax.annotate(f"a (f={Y[0]:.4g})", (0, Y[0]), textcoords="offset points", xytext=(8, 8))
        ax.annotate(f"b (f={Y[-1]:.4g})", (1, Y[-1]), textcoords="offset points", xytext=(-60, 8))
        ax.set_xlabel("t  (a + t·(b−a))")
        ax.set_ylabel("f")
        ax.set_title(f"{name} (n={n}) — parametric cross-section\n"
                     f"a={np.array(a).round(3).tolist()}  →  b={np.array(b).round(3).tolist()}")
        ax.legend()
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return (f"min on path: {Y[k]:.6g} at t={t[k]:.4f} "
                f"(f(a)={Y[0]:.6g}, f(b)={Y[-1]:.6g})", None)

    def forward(self, function: str, output_filename: str = None, mode: str = "surface",
                n: int = 2, dims=None, base_point=None, base_is_minimizer=False,
                path_from=None, path_to=None, grid: int = 101, window=None,
                levels: int = None):
        mode = (mode or "surface").lower()
        if mode not in ("surface", "slice", "contour", "path"):
            return f"Error: unknown mode '{mode}' (use surface | slice | contour | path)."
        if not output_filename:
            output_filename = (f"bp_{function.strip().lower().replace(' ', '_')}"
                               f"_{mode}.png")
        elif not output_filename.lower().endswith(".png"):
            return f"Error: output_filename '{output_filename}' must end in .png."
        try:
            grid = max(11, min(int(grid or 101), 401))
            entry, err = self._get_entry(function, mode, n or 2)
            if err:
                return err
            out_png = os.path.join(self.safe_dir, os.path.basename(output_filename))
            if mode == "path":
                detail, err = self._path(function, entry, n or 2, path_from, path_to, grid, out_png)
                if err:
                    return err
            else:
                kind = "slice" if (mode == "slice" or (mode == "contour" and (n or 2) > 2)) else "surface"
                X, Y, Z, xlabel, ylabel, base_note, err = self._build_grid(
                    entry, mode, n or 2, dims or [0, 1], base_point,
                    base_is_minimizer, grid, window)
                if err:
                    return err
                if mode == "contour":
                    detail = self._contour(function, entry, kind, n or 2, dims or [0, 1],
                                           grid, out_png, X, Y, Z, xlabel, ylabel, base_note,
                                           levels)
                else:
                    detail = self._two_panel(function, entry, kind, n or 2, dims or [0, 1],
                                             grid, out_png, X, Y, Z, xlabel, ylabel, base_note,
                                             levels)
            return (f"Plot saved to: {out_png}\nmode={mode} function={function} "
                    f"grid={grid}\n{detail}")
        except Exception as e:
            return f"Plot error: {type(e).__name__}: {e}"
