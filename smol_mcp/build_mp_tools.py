#!/usr/bin/env python3
"""build_mp_tools.py — generate smol_mcp/tools_lib_mp.py + vendor/ for the
Phase-2 mpmath (MP) high-precision optimizers. Re-runnable and idempotent.

What it does
  1. Extracts 5 classes from the READ-ONLY monolith via AST (exact,
     non-guessed boundaries):
         SimAnnealToolMP, GAToolMP, DifferentialEvolutionMP,
         IslandModelDEMP, DEToolMP
  2. Fixes the broken mpmath precision idiom:
         `mp.dps = X`  /  `mp.dps`   ->   `mp.mp.dps`
     Module-level `mp.dps` is a SILENT NO-OP on mpmath<=1.3 (context stays at
     dps=15) and a hard AttributeError on mpmath>=1.4 (guard added for
     mpmath Issue #657). The context-level `mp.mp.dps` is the correct,
     version-agnostic setter — verified to set real precision on 1.4.1.
     The regex is idempotent (negative lookbehind) so re-running is safe.
  3. Vendors the external optimizer engines into ./vendor so smol_mcp is
     FULLY SELF-CONTAINED (no runtime dependency on the source repo):
         core.py, neighbors.py, genetic.py, schedules.py   (standard SA/GA)
         core_mp_gpt5.py, genetic_mp.py                     (MP engines, fixed)
  4. Applies loader fixes (fix_loader, idempotent):
         a. Bulletproof dict bounds stored as a string (all *_mp variants)
            are parsed with ast.literal_eval instead of fed raw to mp.mpf.
         b. numpy-style bulletproof functions execute against an mpmath
            shim bound to 'np' (numpy ufuncs raise TypeError on mpf);
            `import numpy as np` lines are stripped from the code strings.
         c. DEToolMP CSV header uses len(bounds_list) (len(bounds) crashed
            when the caller omitted bounds).

Run:  .venv/Scripts/python.exe build_mp_tools.py
"""
import ast, io, os, re, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
MONOLITH = os.environ.get(
    "SMOL_MONOLITH",
    r"C:\Users\MATTIA\AppData\Local\hermes\attachments\MyGsearchSmolAgentsZ0WEM_OpAI_SWI_V04_i_SA_s_n_GA_BO_Gv16.py",
)
REPO = os.environ.get("SMOL_MCP_REPO_DIR", r"D:\Source\Repos\MyGSearchSmolagentsEM")

# Extraction order = definition order in the monolith.
MP_CLASSES = ["SimAnnealToolMP", "GAToolMP", "DifferentialEvolutionMP",
              "IslandModelDEMP", "DEToolMP"]
VENDOR_PLAIN = ["core.py", "neighbors.py", "genetic.py", "schedules.py"]
VENDOR_MP = ["core_mp_gpt5.py", "genetic_mp.py"]  # these get the dps fix

# `mp.dps` -> `mp.mp.dps`, but NOT if already `mp.mp.dps` (idempotent).
DPS_RE = re.compile(r"(?<!mp\.)mp\.dps\b")


def fix_dps(text: str):
    fixed, n = DPS_RE.subn("mp.mp.dps", text)
    return fixed, n


# ── Loader fixes (applied to each extracted class segment, idempotent) ──
# The monolith's bulletproof loader has three bugs for MP execution:
#   1. *_mp variants store bounds as a STRING; the loader passed that raw
#      string to mp.mpf -> "invalid literal for int() with base 10: '['".
#   2. numpy-style bulletproof functions (base variants) call np.cos etc.
#      on mpf scalars -> "loop of ufunc does not support argument 0 of
#      type mpf". Fixed by binding an mpmath shim as 'np' in the exec
#      namespace and stripping the code's own `import numpy as np`.
#   3. DEToolMP built its CSV header with len(bounds); bounds is None when
#      the caller omitted it -> TypeError.
LOADER_FIXES = [
    (
        '            if not bounds:\n'
        '                bounds_list = data["bounds"]\n',
        '            if not bounds:\n'
        '                # Loader fix: bulletproof dict stores some bounds as\n'
        '                # a string (all *_mp variants) - parse them like\n'
        '                # user-supplied strings before mp.mpf conversion.\n'
        '                raw_bounds = data["bounds"]\n'
        '                if isinstance(raw_bounds, str):\n'
        '                    raw_bounds = ast.literal_eval(raw_bounds)\n'
        '                bounds_list = raw_bounds\n',
    ),
    (
        "global_scope = {'mp': mp, 'math': mp}",
        "global_scope = {'mp': mp, 'math': mp, 'np': _mp_numpy_shim}",
    ),
    (
        "namespace = {'mp': mp, 'np': __import__('numpy')}",
        "namespace = {'mp': mp, 'np': _mp_numpy_shim}",
    ),
    (
        "exec(function_code, global_scope, local_scope)",
        "exec(_prepare_bulletproof_code(function_code), global_scope, local_scope)",
    ),
    (
        "exec(code_to_compile, global_scope, local_scope)",
        "exec(_prepare_bulletproof_code(code_to_compile), global_scope, local_scope)",
    ),
    (
        "exec(code_to_compile, namespace)",
        "exec(_prepare_bulletproof_code(code_to_compile), namespace)",
    ),
    (
        "range(len(bounds))",
        "range(len(bounds_list))",
    ),
]


def fix_loader(text: str):
    fixed, n = text, 0
    for old, new in LOADER_FIXES:
        c = fixed.count(old)
        if c:
            fixed = fixed.replace(old, new)
            n += c
    return fixed, n


def extract_classes(monolith_path: str) -> dict:
    src = io.open(monolith_path, encoding="utf-8").read()
    lines = src.splitlines()
    tree = ast.parse(src)
    classes = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in MP_CLASSES:
            seg = "\n".join(lines[node.lineno - 1:node.end_lineno])
            classes[node.name] = (node.lineno, node.end_lineno, seg)
    missing = [c for c in MP_CLASSES if c not in classes]
    if missing:
        raise SystemExit(f"ERROR: classes not found in monolith: {missing}")
    return classes


def main():
    print(f"monolith: {MONOLITH}")
    classes = extract_classes(MONOLITH)

    # ---- 1. build tools_lib_mp.py ----
    total = 0
    total_loader = 0
    blocks = []
    for name in MP_CLASSES:
        lo, hi, seg = classes[name]
        fixed, n = fix_dps(seg)
        fixed, n2 = fix_loader(fixed)
        total += n
        total_loader += n2
        blocks.append(fixed)
        print(f"  extracted {name:22s} L{lo}-{hi} ({len(fixed.splitlines())} lines, {n} dps, {n2} loader fixes)")

    header = '''#!/usr/bin/env python3
"""tools_lib_mp.py — Phase-2 mpmath (MP) high-precision optimizers.

GENERATED by build_mp_tools.py from the read-only monolith. Do not edit by
hand; re-run the builder. The mpmath precision idiom was fixed to the
context-level `mp.mp.dps` (module-level `mp.dps` never actually set precision).
Loader fixes (string bounds parsing, numpy→mpmath shim, CSV header) are
applied by build_mp_tools.py::fix_loader — see the builder for details.

External MP engines (core_mp_gpt5, genetic_mp) are vendored in ./vendor and
resolved via sys.path by mcp_server.py.
"""
from __future__ import annotations
import os, ast, csv, logging, random, re
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use("Agg")  # headless MCP: plots are saved to files, never shown
import matplotlib.pyplot as plt
from smolagents.tools import Tool
from tools_lib import BULLETPROOF_FUNCTIONS, csv_field_delim

# ─────────────────────────────────────────────────────────────────────────
# numpy→mpmath compatibility shim (loader fix, applied by the builder)
#
# Bulletproof functions written for numpy (np.sin, np.cos, ...) break when
# executed with mpmath mpf scalars: numpy ufuncs raise
# "TypeError: loop of ufunc does not support argument 0 of type mpf".
# The shim rebinds math ufuncs/constants to mpmath (full precision) and
# falls back to real numpy for anything else. It is bound as 'np' in the
# exec namespace; the `import numpy as np` lines in the code strings are
# stripped first (they would shadow the shim).
# ─────────────────────────────────────────────────────────────────────────
_NUMPY_IMPORT_RE = re.compile(
    r"^\\s*import\\s+numpy\\s*(?:as\\s+\\w+)?\\s*$", re.MULTILINE
)


class _MPShimNDArray:
    """Stand-in for np.ndarray: isinstance() accepts the class, and the MP
    engines' plain-list inputs are never instances of it."""


class _MPShimRandom:
    """Stand-in for np.random: bulletproof code (mogadish) only calls
    default_rng(seed); the RNG object is created but never consumed (the
    'chaos' is a deterministic function of the seed)."""

    @staticmethod
    def default_rng(seed=0):
        return object()


class _MPNumpyShim:
    """numpy facade that routes scalar math to mpmath (see header note)."""

    _UFUNC_MAP = {
        'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
        'arcsin': 'asin', 'arccos': 'acos',
        'arctan': 'atan', 'arctan2': 'atan2',
        'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh',
        'exp': 'exp', 'expm1': 'expm1',
        'log': 'log', 'log10': 'log10', 'log1p': 'log1p',
        'sqrt': 'sqrt', 'abs': 'fabs', 'fabs': 'fabs',
        'floor': 'floor', 'ceil': 'ceil',
    }

    def __getattr__(self, name):
        if name == 'round':
            # mpmath has no module/context-level round; Python's built-in
            # round() on mpf is banker's rounding, identical to np.round
            return round
        if name in self._UFUNC_MAP:
            return getattr(mp, self._UFUNC_MAP[name])
        if name in ('pi', 'e', 'inf', 'nan'):
            return getattr(mp, name)
        if name == 'ndarray':
            return _MPShimNDArray
        if name == 'random':
            return _MPShimRandom
        return getattr(np, name)  # everything else: real numpy


_mp_numpy_shim = _MPNumpyShim()


def _prepare_bulletproof_code(code: str) -> str:
    """Strip `import numpy [as np]` lines so the shim bound to 'np' in the
    exec namespace is what the compiled function actually sees."""
    return _NUMPY_IMPORT_RE.sub("", code)
'''
    out = os.path.join(HERE, "tools_lib_mp.py")
    with io.open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n\n" + "\n\n".join(blocks) + "\n")
    print(f"\n  wrote {os.path.basename(out)}  ({total} dps fixes, {total_loader} loader fixes across the 5 classes)")

    # ---- 2. vendor the engines ----
    vdir = os.path.join(HERE, "vendor")
    os.makedirs(vdir, exist_ok=True)
    for fn in VENDOR_PLAIN:
        shutil.copyfile(os.path.join(REPO, fn), os.path.join(vdir, fn))
        print(f"  vendored {fn} (as-is)")
    for fn in VENDOR_MP:
        text = io.open(os.path.join(REPO, fn), encoding="utf-8").read()
        fixed, n = fix_dps(text)
        io.open(os.path.join(vdir, fn), "w", encoding="utf-8", newline="\n").write(fixed)
        print(f"  vendored {fn} ({n} dps fix)")

    print("\nDONE — tools_lib_mp.py + vendor/ ready.")


if __name__ == "__main__":
    main()
