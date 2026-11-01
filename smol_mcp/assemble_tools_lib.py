#!/usr/bin/env python3
"""Assemble smol_mcp/tools_lib.py from the original monolith (READ-ONLY).

Dead-code elimination: start from the chosen Phase-1 tool classes, take the
transitive closure over the top-level helper functions and constants they
reference, then emit a clean module containing ONLY:
  - the import statements actually used by the included code
  - the included class definitions (tools + helper classes + base classes)
  - the included helper function definitions
  - the included safe module-level constants
Every top-level *executable* statement (side effects: network, headless Chrome,
matplotlib.use, logging.basicConfig, tool/agent/gradio setup) is excluded.

The original file is never modified.
"""
import ast, os, sys

SRC = r"C:\Users\MATTIA\AppData\Local\hermes\attachments\MyGsearchSmolAgentsZ0WEM_OpAI_SWI_V04_i_SA_s_n_GA_BO_Gv16.py"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools_lib.py")

# The tools to expose in Phase 1 (self-contained or safe_dir-only).
# EXCLUDED by user decision: mem0/ChromaDB tools, MarkdownToPDFTool (md2pdf),
# SearXNG and all web-search tools (MyGoogleSearchTool, SearXNGSearchTool),
# RVGTool (mpmath incompatibility — skipped, not fixed), SafeOpenTool (not
# exposed as a tool; it remains in tools_lib.py as the base class of the
# CSV/PDF tools, pulled in automatically by the closure).
PHASE1_TOOLS = [
    "LCSAnalyzerTool",
    "CSV2TUPTool",
    "PDF2TXTTool",
    "PDFAnalyzerTool",
    "OCR2TXTTool",
    "PDFSplitterToolGv2",
    "ListBPFunctionsTool",
    "GetBPFunctionCodeTool",
    "SimAnnealTool",
    "SAAnalysisTool",
    "GATool",
    "GAAnalysisTool",
    "BOToolGv1_v3_2_0",
    "BOAnalysisTool",
]

# Top-level constants the tools read via globals()['...'] (string lookups the
# AST closure cannot detect). Force-include them so behavior matches the original.
FORCED_CONSTS = ["BULLETPROOF_FUNCTIONS", "csv_field_delim"]

with open(SRC, encoding="utf-8") as f:
    text = f.read()
tree = ast.parse(text)

top_funcs, top_classes, top_consts, import_stmts = {}, {}, {}, []
for n in tree.body:
    if isinstance(n, (ast.Import, ast.ImportFrom)):
        import_stmts.append(n)
    elif isinstance(n, ast.FunctionDef):
        top_funcs[n.name] = n
    elif isinstance(n, ast.ClassDef):
        top_classes[n.name] = n
    elif isinstance(n, ast.Assign):
        for t in n.targets:
            if isinstance(t, ast.Name):
                top_consts[t.id] = n
    elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
        top_consts[n.target.id] = n

def src_of(node):
    return ast.get_source_segment(text, node) or ""

def names_loaded(node):
    ids = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            ids.add(sub.id)
    return ids

def base_names(node):
    out = []
    for b in node.bases:
        if isinstance(b, ast.Name):
            out.append(b.id)
        elif isinstance(b, ast.Attribute):
            out.append(b.attr)
    return out

_SAFE_CALL_ATTRS = {
    "compile", "join", "path", "dirname", "abspath", "basename", "time",
    "randint", "choice", "get", "str", "int", "float", "list", "dict", "set",
    "tuple", "enumerate", "range", "len", "append", "min", "max", "abs", "round",
}

def const_is_safe(node):
    val = node.value
    if val is None:
        return False
    for sub in ast.walk(val):
        if isinstance(sub, ast.Call):
            f = sub.func
            attr = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "?")
            if attr not in _SAFE_CALL_ATTRS:
                return False
    return True

# ---- transitive closure ----
included_classes = {c for c in PHASE1_TOOLS if c in top_classes}
missing = set(PHASE1_TOOLS) - included_classes
if missing:
    print(f"WARNING: these requested tools were not found as top-level classes: {missing}", file=sys.stderr)

included_funcs = set()
included_consts = {c for c in FORCED_CONSTS if c in top_consts}
unsafe_refs = set()

changed = True
while changed:
    changed = False
    for c in list(included_classes):
        for b in base_names(top_classes[c]):
            if b in top_classes and b not in included_classes:
                included_classes.add(b); changed = True
    ref = set()
    for c in included_classes:
        ref |= names_loaded(top_classes[c])
    for fn in included_funcs:
        ref |= names_loaded(top_funcs[fn])
    for name in ref:
        if name in top_funcs and name not in included_funcs:
            included_funcs.add(name); changed = True
    for name in ref:
        if name in top_classes and name not in included_classes:
            included_classes.add(name); changed = True
    for name in ref:
        if name in top_consts and name not in included_consts:
            if const_is_safe(top_consts[name]):
                included_consts.add(name); changed = True
            else:
                unsafe_refs.add(name)

# Imports that pull native deps unavailable on this host (WeasyPrint needs GTK,
# helium/soundfile are only used by Phase-1.5/excluded tools). Keep them out of
# the top-level import block; a lazy shim (see LAZY_MODULES) defers them to use.
EXCLUDED_IMPORT_MODULES = {"md2pdf", "helium", "soundfile"}

# ---- needed imports (AST-based: only names actually used in the included code) ----
included_src = "\n".join(
    [src_of(top_classes[c]) for c in sorted(included_classes)] +
    [src_of(top_funcs[f]) for f in sorted(included_funcs)] +
    [src_of(top_consts[c]) for c in sorted(included_consts)]
)
used_names = {n.id for n in ast.walk(ast.parse(included_src)) if isinstance(n, ast.Name)}

def import_ids(stmt):
    ids = set()
    if isinstance(stmt, ast.Import):
        for a in stmt.names:
            ids.add((a.asname or a.name).split(".")[0])
    elif isinstance(stmt, ast.ImportFrom):
        for a in stmt.names:
            if a.name != "*":
                ids.add(a.asname or a.name)
    return ids

def import_module(stmt):
    if isinstance(stmt, ast.Import):
        return stmt.names[0].name.split(".")[0]
    if isinstance(stmt, ast.ImportFrom) and stmt.module:
        return stmt.module.split(".")[0]
    return None

needed_imports = []
for s in import_stmts:
    if import_module(s) in EXCLUDED_IMPORT_MODULES:
        continue
    if any(i in used_names for i in import_ids(s)):
        needed_imports.append(s)

# ---- emit ----
HEADER = '''"""Auto-generated clean tool library for the smol-mcp server. DO NOT EDIT BY HAND.

Generated by assemble_tools_lib.py from
MyGsearchSmolAgentsZ0WEM_OpAI_SWI_V04_i_SA_s_n_GA_BO_Gv16.py (read-only).
Contains only the Phase-1 smolagents Tool classes, the helper classes/functions
and constants they reference, and the imports they use. All top-level side
effects (network, headless Chrome, matplotlib.use, logging setup, agent/gradio
wiring) are excluded.
"""
from __future__ import annotations
import logging

logger = logging.getLogger("smol_mcp.tools")
'''

parts = [HEADER]
# imports (original source, preserving style)
for s in needed_imports:
    parts.append(src_of(s))
# Forced imports: in the monolith these are nested in a try-block ("dependency
# check"), so the top-level AST scan misses them, but BOToolGv1_v3_2_0 needs them.
FORCED_IMPORTS = [
    "from bayes_opt import BayesianOptimization",
    "from bayes_opt.acquisition import UpperConfidenceBound, ExpectedImprovement, ProbabilityOfImprovement",
]
for line in FORCED_IMPORTS:
    if not any(line in src_of(s) for s in needed_imports):
        parts.append(line)
parts.append("")
# constants first (functions/classes may reference them at definition time)
for c in sorted(included_consts):
    parts.append(src_of(top_consts[c]))
    parts.append("")
# helper classes before tool classes (base classes / referenced classes)
for c in sorted(included_classes, key=lambda n: top_classes[n].lineno):
    parts.append(src_of(top_classes[c]))
    parts.append("")
# functions
for f in sorted(included_funcs, key=lambda n: top_funcs[n].lineno):
    parts.append(src_of(top_funcs[f]))
    parts.append("")

out_text = "\n".join(parts)

# ---- post-generation data fix (idempotent) ----
# The monolith's langermann entries carry the literature value of the
# CLASSICAL Langermann function (-5.147 at (8.20, 9.21)), but the dict's
# code is a different 5-term Gaussian-cosine sum. Its verified global min
# (dps=60; gradient < 1e-12 at the minimizer; every other well <= -3.0;
# full-domain grid scan confirms no deeper basin) is:
#   -5.1621261599639824  at (2.0029921199344658384, 1.0060959402921884107)
n_fix = out_text.count('"global_min": -5.147')
out_text = out_text.replace(
    '"global_min": -5.147',
    '"global_min": -5.1621261599639824  # verified 2026-10: true min of the 5-term sum; the lit. -5.147 belongs to the classical Langermann',
)

# ---- post-generation data fix: expose BO's random_state as `seed` (idempotent) ----
# The monolith hardcodes random_state=1 in BOToolGv1_v3_2_0, making BO runs
# reproducible but single-shot. Expose it as a `seed` parameter: default 1
# preserves every previously documented result exactly; a different value
# (or None) enables true multi-start.
if "random_state=seed" not in out_text:
    out_text = out_text.replace(
        "acq_type: str = 'ucb', kappa: float = 2.5, xi: float = 0.0):",
        "acq_type: str = 'ucb', kappa: float = 2.5, xi: float = 0.0,\n"
        "                seed: int = 1):",
        1)
    out_text = out_text.replace(
        '''        "xi": {
            "type": "number",
            "description": "Exploration weight for EI. Default 0.0.",
            "nullable": True
        }
    }''',
        '''        "xi": {
            "type": "number",
            "description": "Exploration weight for EI. Default 0.0.",
            "nullable": True
        },
        "seed": {
            "type": "integer",
            "description": "Random seed for initial-point sampling. Default 1 (reproducible). Pass a different value, or null for a non-deterministic run (multi-start).",
            "nullable": True
        }
    }''',
        1)
    out_text = out_text.replace("random_state=1,", "random_state=seed,", 1)

# ---- post-generation data fix: analyzer hardening (idempotent) ----
# 1) SAAnalysisTool: accept the MP SA CSV format (no best_so_far / accepted
#    columns) — derive the running best from energy, skip the acceptance
#    section when 'accepted' is not logged.
# 2) SAAnalysisTool: the adaptive-scaling check compared a PERCENT value
#    (late_accept, e.g. 45.2) against a fraction (0.3..0.6), making the
#    "worked perfectly" branch unreachable. Now compares 30..60 (%).
# 3) BOAnalysisTool: for n<=10 the else-branch mislabeled flat runs as
#    "LEARNING: Improved by 0.0000"; now reports the check as skipped.
if "Convergence check skipped" not in out_text:
    _FIXES = [
        # (1) SA parse block: best_so_far fallback + optional accepted
        ("""            best_energy = np.array([float(r['best_so_far']) for r in rows])  # recommended to log
            accepted = np.array([int(r.get('accepted', 1)) for r in rows])  # fallback 1""",
         """            # best_so_far is logged by the numpy SA tool; the MP variant only
            # logs per-iteration energy, so derive the running best from it.
            if rows[0].get('best_so_far') is not None:
                best_energy = np.array([float(r['best_so_far']) for r in rows])
            else:
                best_energy = np.minimum.accumulate(energy)
            # 'accepted' is only logged by the numpy SA tool; without it the
            # acceptance section is skipped (MP variant).
            has_accepted = rows[0].get('accepted') is not None
            accepted = (np.array([int(r['accepted']) for r in rows])
                        if has_accepted else None)"""),
        # (2) SA acceptance section: guard for missing 'accepted'
        ("""            # 1. Acceptance Rate Analysis
            accept_rate = accepted.mean() * 100
            late_accept = accepted[-max(100, n//10):].mean() * 100
            report += [
                "ACCEPTANCE RATE",
                "-"*30,
                f"Overall acceptance     : {accept_rate:.1f}%",
                f"Late-phase acceptance  : {late_accept:.1f}%",
            ]
            if accept_rate < 15:
                report += ["Too low! Steps too small or temperature too low early."]
                report += ["→ Enable or keep adapt_neighbor=True"]
                report += ["→ Try higher initial_temperature"]
            elif accept_rate > 70:
                report += ["Too high! Too random, wasting evaluations."]
                report += ["→ Lower initial_temperature"]
                report += ["→ Use faster cooling (e.g. cooling_rate=0.90)"]
            else:
                report += ["Healthy range (15–70%). Good exploration/exploitation balance."]""",
         """            # 1. Acceptance Rate Analysis
            if accepted is not None:
                accept_rate = accepted.mean() * 100
                late_accept = accepted[-max(100, n//10):].mean() * 100
                report += [
                    "ACCEPTANCE RATE",
                    "-"*30,
                    f"Overall acceptance     : {accept_rate:.1f}%",
                    f"Late-phase acceptance  : {late_accept:.1f}%",
                ]
                if accept_rate < 15:
                    report += ["Too low! Steps too small or temperature too low early."]
                    report += ["→ Enable or keep adapt_neighbor=True"]
                    report += ["→ Try higher initial_temperature"]
                elif accept_rate > 70:
                    report += ["Too high! Too random, wasting evaluations."]
                    report += ["→ Lower initial_temperature"]
                    report += ["→ Use faster cooling (e.g. cooling_rate=0.90)"]
                else:
                    report += ["Healthy range (15–70%). Good exploration/exploitation balance."]
            else:
                accept_rate = None
                late_accept = None
                report += [
                    "ACCEPTANCE RATE",
                    "-"*30,
                    "Not logged by this engine (MP variant) - section skipped.",
                ]"""),
        # (3) SA adaptive scaling: percent fix + None guard
        ("""                if 0.3 < late_accept < 0.6:
                    report += ["Adaptive scaling worked perfectly!"]""",
         """                if late_accept is None:
                    report += ["Scale varied, but acceptance not logged - effectiveness unknown."]
                elif 30 < late_accept < 60:
                    report += ["Adaptive scaling worked perfectly!"]"""),
        # (4) SA action plan: guard None accept_rate
        ("""            if accept_rate > 70:
                recs += ["Lower initial_temperature by 50%"]""",
         """            if accept_rate is not None and accept_rate > 70:
                recs += ["Lower initial_temperature by 50%"]"""),
        # (5) BO convergence: honest n<=10 message instead of LEARNING 0.0000
        ("""            if improvement_late <= 0 and n > 10:
                report += ["⚠️ STALLED: No improvement in second half."]
                report += ["   -> HINT: You might be exploiting too much. Switch to 'simulated_annealing' to fine-tune."]
            else:
                report += [f"✅ LEARNING: Improved by {improvement_late:.4f} in later stages."]""",
         """            if improvement_late <= 0:
                if n > 10:
                    report += ["⚠️ STALLED: No improvement in second half."]
                    report += ["   -> HINT: You might be exploiting too much. Switch to 'simulated_annealing' to fine-tune."]
                else:
                    report += ["ℹ️ Convergence check skipped: not enough samples (n<=10)."]
            else:
                report += [f"✅ LEARNING: Improved by {improvement_late:.4f} in later stages."]"""),
    ]
    for _old, _new in _FIXES:
        if _old not in out_text:
            raise SystemExit(
                "analyzer-hardening fix: expected text not found in generated "
                "output (monolith drift?):\n" + _old[:120])
        out_text = out_text.replace(_old, _new, 1)
    print("  fix      : 5 analyzer hardening replacements applied "
          "(SA MP-format support, adaptive-% fix, BO n<=10 message)")

# ---- post-generation merge: project-owned extra bulletproof functions ----
# The monolith is read-only; new benchmark entries (added after the monolith
# was frozen) live in bulletproof_extra.py with the same schema and are merged
# into the generated BULLETPROOF_FUNCTIONS here. Idempotent: names already
# present in the monolith copy are skipped.
EXTRA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bulletproof_extra.py")
n_extra = 0
if os.path.exists(EXTRA_PATH):
    import importlib.util
    spec = importlib.util.spec_from_file_location("bulletproof_extra", EXTRA_PATH)
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    gen_tree = ast.parse(out_text)
    bp_assign = next(
        n for n in gen_tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "BULLETPROOF_FUNCTIONS" for t in n.targets)
    )
    # every string constant in the dict (the 34 keys + their code strings);
    # a function name can never collide with a code string
    existing_keys = {
        c.value for c in ast.walk(bp_assign.value)
        if isinstance(c, ast.Constant) and isinstance(c.value, str)
    }

    def fmt_entry(name, data):
        code = data["code"].strip("\n")
        gm = repr(data["global_min"])
        if data.get("global_min_note"):
            gm += f"  # {data['global_min_note']}"
        return (
            f'    "{name}": {{\n'
            f'        "code": """{code}""",\n'
            f'        "bounds": {data["bounds"]!r},\n'
            f'        "global_min": {gm}\n'
            f'    }},\n'
        )

    lines = out_text.splitlines(keepends=True)
    new_entries = []
    for name, data in _mod.BULLETPROOF_EXTRA.items():
        if name in existing_keys:
            continue
        new_entries.append(fmt_entry(name, data))
        n_extra += 1
    if new_entries:
        insert_at = bp_assign.end_lineno - 1  # line index of the dict's closing '}'
        # the monolith's last entry has no trailing comma (it was last); the
        # inserted entries need one, so add it to the preceding '}' line
        k = insert_at - 1
        while k >= 0 and lines[k].strip() == "":
            k -= 1
        if k >= 0 and lines[k].rstrip().endswith("}"):
            lines[k] = lines[k].rstrip() + ",\n"
        lines[insert_at:insert_at] = new_entries
        out_text = "".join(lines)

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(out_text)

print(f"Wrote {OUT} ({os.path.getsize(OUT)} bytes)")
if n_fix:
    print(f"  data fix: {n_fix} langermann global_min reference(s) corrected (-5.147 -> -5.1621261599639824)")
if n_extra:
    print(f"  extra    : {n_extra} bulletproof entr(ies) merged from bulletproof_extra.py")
print(f"  imports : {len(needed_imports)}")
print(f"  classes : {len(included_classes)} -> {sorted(included_classes)}")
print(f"  funcs   : {len(included_funcs)} -> {sorted(included_funcs)}")
print(f"  consts  : {len(included_consts)} -> {sorted(included_consts)}")
if unsafe_refs:
    print(f"  WARNING: referenced top-level constants that were NOT included (unsafe value): {sorted(unsafe_refs)}", file=sys.stderr)
