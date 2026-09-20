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

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(parts))

print(f"Wrote {OUT} ({os.path.getsize(OUT)} bytes)")
print(f"  imports : {len(needed_imports)}")
print(f"  classes : {len(included_classes)} -> {sorted(included_classes)}")
print(f"  funcs   : {len(included_funcs)} -> {sorted(included_funcs)}")
print(f"  consts  : {len(included_consts)} -> {sorted(included_consts)}")
if unsafe_refs:
    print(f"  WARNING: referenced top-level constants that were NOT included (unsafe value): {sorted(unsafe_refs)}", file=sys.stderr)
