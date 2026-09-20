# Smolagents-Tools → Hermes MCP Server — Implementation Plan

> **For Hermes:** Use the subagent-driven-development skill to implement this plan task-by-task.

## ✅ EXECUTION STATUS (Phase 1 COMPLETE — 2026-09-19)

The server is built, tested, and registered. Final Phase 1 tool set (14 tools,
after user exclusions):

**Excluded by user decision:** mem0/ChromaDB tools, `MarkdownToPDFTool`
(md2pdf/WeasyPrint needs GTK — unavailable on this host), SearXNG and all
web-search tools, `RVGTool` (mpmath 1.4.1 incompatibility — skipped, not fixed),
`SafeOpenTool` (not exposed as a tool; remains the base class of the CSV/PDF
tools).

**Exposed (14):** `lcs_analyzer`, `csv_2_tup_tool`, `pdf_2_txt_tool`,
`pdf_analyzer_tool`, `ocr_2_txt_tool`, `pdf_splitter_tool`,
`list_bulletproof_functions`, `get_bulletproof_function_code`,
`simulated_annealing_optimizer`, `sa_analysis_tool`,
`genetic_algorithm_optimizer`, `ga_analysis_tool`, `bayesian_optimizer`,
`bo_analysis_tool`.

**Verified:** 3/3 pytest pass; live MCP stdio handshake passes
(`initialize` → `list_tools` (14) → `call_tool` on `list_bulletproof_functions`
and a real `simulated_annealing_optimizer` run (ackley → "Optimization
Successful") → graceful-error call → server stays alive).

**Delivered in `D:\Source\hermes-dir\smol_mcp\`:**
- `requirements.txt` — `smolagents>=1.26,<2` (1.27 does not exist on PyPI;
  max is 1.26.0), `mcp==2.0.0`, plus per-tool deps and `bayesian-optimization`.
- `assemble_tools_lib.py` → `tools_lib.py` — ast dead-code-elimination of 16
  classes + 2 helper funcs + 3 shared constants, no top-level side effects.
- `mcp_server.py` — `MCPServer` + generic wrapper + optimizer-engine injection
  + stdout guards.
- `tests/test_mcp_server.py` (pytest), `tests/test_stdio_handshake.py`
  (live MCP client), `tests/probe_manual.py` (raw JSON-RPC probe).
- `register_config.py` — idempotent `config.yaml` insert (backup saved to
  `~/.hermes/backups/config.yaml.bak-smolmcp-*`).

**Config:** `~/.hermes/config.yaml → mcp_servers.smol-mcp` (stdio; `command`
= the dedicated venv python, `args` = `mcp_server.py`, `env` = data dir +
repo dir). **Restart the Hermes session** for the tools to load.

**Pitfalls hit & fixed (also saved to skill `embed-python-tools-as-mcp-server`):**
1. `fitz` prints a deprecation warning to **stdout** on import → corrupts the
   JSON-RPC channel. Fixed by redirecting stdout→stderr around imports AND
   every `forward()` call.
2. `tool.forward` is a **bound** method (no `self` in its signature) —
   `params[1:]` silently dropped the first real param. Skip only a param
   named `self`.
3. Wrapper annotations are eval'd against `__globals__` — `exec` into module
   globals, not an empty namespace.
4. smolagents `inputs` metadata is often wrong (over-required / over-typed
   params) — detect optionality & polymorphism from the tool source.
5. Optimizer engines (`SimulatedAnnealing`, `sa_neighbors`,
   `RealGeneticAlgorithm`, `BayesianOptimization`+acquisitions) were imported
   in the monolith's `__main__`/`try` blocks → loaded at server startup from
   the local repo modules + `bayes_opt` and bound into `tools_lib`'s namespace.

---

**Goal:** Expose the reusable smolagents tools from
`MyGsearchSmolAgentsZ0WEM_OpAI_SWI_V04_i_SA_s_n_GA_BO_Gv16.py` to Hermes Agent
through a dedicated stdio MCP server, so they become first-class Hermes tools.

**Architecture:** A small self-contained package (`smol_mcp/`) with (1) a clean
`tools_lib.py` that holds *only* the tool class definitions + the helper classes
and constants they need (extracted deterministically from the original monolith,
which is never modified), and (2) an `mcp_server.py` built on the `mcp` 2.0.0
`MCPServer` that turns each smolagents `Tool` into an MCP tool via a generic
dynamic wrapper. Hermes launches it as a stdio MCP server from `config.yaml`.

**Tech Stack:** Python 3.11, `mcp==2.0.0` (`MCPServer`, `@server.tool()`,
`run_stdio_async()`), `smolagents` (the `Tool` base class), plus per-tool deps
(`fitz`/`pymupdf4llm`, `md2pdf`, `pdf2image`+`pytesseract`, `numpy`, `mpmath`,
`nltk`+`spacy`, `httpx`). Dedicated venv.

---

## 1. Context & verified assumptions

Everything below was checked against the live machine before writing this plan.

### 1.1 The source file
- Path: `C:\Users\MATTIA\AppData\Local\hermes\attachments\MyGsearchSmolAgentsZ0WEM_OpAI_SWI_V04_i_SA_s_n_GA_BO_Gv16.py`
- Size: **382,355 bytes / 9,148 lines** — too large to hold in model context.
- It is a **smolagents `CodeAgent` script**, not a library: tools are `Tool`
  subclasses with a `forward()` method, wired into a `tools_list` and a
  `GuardedLocalExecutor` (a code-execution sandbox), then served by Gradio.
- **61 classes** total; **49 are `Tool` subclasses** (the rest are helpers:
  `GuardedLocalExecutor`, `_CohenEngine`, `_AffineEngine`, `LCG`,
  `RandomVariateGenerator`, `_TextProcessor`, `DifferentialEvolutionMP`,
  `IslandModelDEMP`, `VaultJSONEncoder`, `SharedMem0Manager`, two `*LiteLLMModel`).
- **Top-level side effects on import** (why we cannot just `import` the file):
  - `matplotlib.use(...)` (L5270), `logging.basicConfig(...)` (L186)
  - `BULLETPROOF_FUNCTIONS = {...}` built at module level (L5738) — *needed*
  - SearXNG setup: `searxservers` dict (L2551) + a `start_chrome(headless=True)`
    block (L2594–2653) that is **inside a `'''...'''` string** (i.e. inert), but
    surrounding `httpx`/network code and `before = time.time()` are live.
  - Folder-path strings (L8563+) are harmless (`os.path.join` only, no `mkdir`).
- **User standing rule (from profile):** never modify already-saved versions.
  → The original `.py` is **read-only**. All new code goes in `smol_mcp/`.

### 1.2 The MCP runtime (verified)
- `mcp==2.0.0` is installed in the Hermes venv
  (`...\hermes-agent\venv\Lib\site-packages\mcp`).
- `mcp.server.fastmcp` **no longer exists** in 2.0.0. The working API (documented
  in Hermes's own `hermes-agent/mcp_serve.py`) is:
  ```python
  from mcp.server import MCPServer
  server = MCPServer("name", instructions="...")
  @server.tool()                 # docstring -> description, signature -> input schema
  def my_tool(a: str, b: int = 5) -> str: ...
  await server.run_stdio_async()
  ```
- Verified: `@server.tool()` **auto-derives a correct JSON Schema** from a
  typed signature, and `required` lists only the params with no default. The
  `Tool` object returned by `await server.list_tools()` exposes the schema as
  `tool.input_schema` (snake_case, **not** `inputSchema`).
- Verified: `@server.tool()` works on a **dynamically-constructed function**
  (one whose `__annotations__`, `__defaults__` and docstring are set programmatically)
  — this is what the generic wrapper relies on.

### 1.3 Hermes MCP registration (verified)
- config.yaml → mcp_servers: supports **both** transports. From
  `tools/mcp_tool_discovery.py`: transport = `"http" if "url" in cfg else "stdio"`;
  a stdio entry is read via `new_servers.get(name, {}).get("command")`.
- Confirmed in `acp_adapter/server.py`: `McpServerStdio` → `{"command", "args", "env"}`;
  `McpServerHttp` → `{"url", "headers"}`.
- Existing entries in this install are all HTTP (`obscura-mcp`, `wolfram`,
  `mcp-wsEM-1.3`), so a stdio entry is new but fully supported.

### 1.4 Environment gaps (verified)
- The **default** `python` (3.11.15) does **not** have `smolagents`, `litellm`,
  or `fastmcp`. `mcp` 2.0.0 **is** present.
- → The MCP server needs its **own venv** with `smolagents` + `mcp` + the
  per-tool deps installed. (The tool classes subclass `smolagents.tools.Tool`,
  so `smolagents` is mandatory even though we only use its `Tool` base.)

### 1.5 Per-tool audit (from `audit.json`)
- **Self-contained / `safe_dir` only (Phase 1 sweet spot):**
  `RVGTool`, `LCSAnalyzerTool`, `SafeOpenTool`, `CSV2TUPTool`,
  `PDF2TXTTool`, `PDFAnalyzerTool`, `OCR2TXTTool`, `PDFSplitterToolGv2`,
  `MarkdownToPDFTool`, `SimAnnealTool`, `SAAnalysisTool`, `GATool`, `GAAnalysisTool`,
  `BOToolGv1_v3_2_0`, `BOAnalysisTool`, `ListBPFunctionsTool`, `GetBPFunctionCodeTool`.
- **`exec()` on arbitrary `function_code` (Phase 2, guard required):**
  `SimAnnealToolMP`, `GAToolMP`, `DEToolMP` (and the standard optimizers accept
  either a named BULLETPROOF function *or* raw code — raw code path uses `exec`).
- **LLM-wrapping (needs `litellm` + `llm_tool_config`):**
  `LLM_Tool`, `Summarization_Tool`, `Translation_Tool`.
- **mem0/ChromaDB (needs `mem0` + `mem0_tool_config`):**
  `CreateChromaDBTool`, `PopulateChromaDBTool`, `QueryChromaDBTool`,
  `ManageChromaDBTool`, `AnalyzeChromaDBTool`.
- **Stateful DSP/HDF5 chain (returns numpy arrays; least MCP-friendly):**
  `AudioHandlerTool`, `TFDTool`, `TFDDataTool`, `LogFreqPlotTool`, `EMDToolGv2`,
  `ArtifactVaultToolGv16`, `ArtifactVaultInspectorToolGv3`,
  `ArtifactVaultByPatternToolGv1`, `SlamFssSifterToolGv3`, `PulseMatchToolV1`,
  `SyncAlignmentToolV2`. Recommend **leaving in smolagents**; optionally expose
  read-only `artifact_inspector` in Phase 3.
- **Web search (network + external lib):** `MyGoogleSearchTool`
  (`engine='duckduckgo'`), `SearXNGSearchTool`.
- **Shared module-level constants the tools read via `globals()`:**
  `BULLETPROOF_FUNCTIONS`, `csv_field_delim` (= `";"`). These **must** exist in
  `tools_lib.py`'s namespace, or the optimizer/BP tools break at runtime.
- **Constructor dependency graph (instantiate once at server startup, mirroring
  the original `tools_list`, L8627–8684):**
  - `SafeOpenTool(safe_dir=...)`, `PDF*Tool(safe_dir=...)`, `CSV2TUPTool(safe_dir=...)`,
    `MarkdownToPDFTool(safe_dir=...)`, `OCR2TXTTool(safe_dir=...)`,
    `PDFAnalyzerTool(safe_dir=...)`
  - Optimizers: `SimAnnealTool(safe_dir=...)`, etc.
  - `SlamFssSifterToolGv3(emd_tool_instance, vault)`, `PulseMatchToolV1(vault)`,
    `SyncAlignmentToolV2(vault)`, `vault = ArtifactVaultToolGv16(safe_dir=..., inspector_tool=scout)`.

---

## 2. What is already extracted (build on this)

From the analysis pass, these artifacts already exist in
`D:\Source\hermes-dir\smolagent_tools_extract\`:

| File | Contents |
|------|----------|
| `manifest.json` | All 49 `Tool` classes: `name`, `name_attr`, `description`, `inputs[]`, `output_type`, `line_start/end`, `bases`, `source_file`. |
| `tools/<ClassName>.py` | Full source of each tool class (49 files), readable in isolation. |
| `audit.json` | Per-tool flags: `exec_eval`, `globals`, `env`, `network`, `init_params`, `heavy_deps`. |
| `schemas.json` | Proof-of-concept `Tool.inputs` → MCP `inputSchema` for all 49. |
| `extract_smaltools.py`, `audit_smaltools.py`, `gen_schemas.py` | The generators (reproducible). |

The plan's Task 1 reuses `extract_smaltools.py`'s `ast` approach to assemble a
clean `tools_lib.py` (tools + helpers + constants, side-effects excluded).

---

## 3. Target file layout

```
D:\Source\hermes-dir\smol_mcp\
├── .venv\                     # dedicated venv (Task 0)
├── tools_lib.py               # ASSEMBLED — tool classes + helpers + constants (Task 1)
├── mcp_server.py              # MCPServer + generic wrapper + startup wiring (Task 2)
├── assemble_tools_lib.py      # deterministic generator for tools_lib.py (Task 1)
├── requirements.txt           # mcp + smolagents + per-tool deps (Task 0)
└── tests\
    └── test_mcp_server.py     # list_tools / call_tool smoke tests (Task 3)
```

Hermes config change (Task 4):
```
C:\Users\MATTIA\AppData\Local\hermes\config.yaml  →  mcp_servers:  (add smol-mcp)
```

---

## 4. Phased plan

### Phase 0 — venv + dependencies
Goal: a venv that can import `smolagents`, `mcp`, and every Phase-1 tool dep.

#### Task 0.1: Create venv + requirements

**Files:** Create `smol_mcp/requirements.txt`

```txt
mcp==2.0.0
smolagents>=1.27
# PDF / docs
PyMuPDF>=1.24          # provides 'fitz'
pymupdf4llm
Pillow
pdf2image
pytesseract
md2pdf
# data / numerics
numpy
mpmath
# search
httpx
# LCS
nltk
spacy
```

**Step 1:** Write the file (content above).

**Step 2:** Create the venv and install (foreground, generous timeout):
```bash
cd D:/Source/hermes-dir/smol_mcp
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
```
Expected: all packages install. **Note:** `pdf2image` needs the **Tesseract
binary** on PATH and `spacy` needs `python -m spacy download en_core_web_sm`
for `LCSAnalyzerTool` — both are optional for the *first* end-to-end tool
(RVG) and can be added when those specific tools are enabled.

**Step 3:** Verify imports:
```bash
.venv/Scripts/python.exe -c "import smolagents, mcp, fitz, numpy, mpmath; print('ok', smolagents.__version__)"
```
Expected: `ok <version>`. (If `fitz`/`spacy` fail, that only blocks those tools;
RVG does not need them.)

**Step 4:** Commit (if `smol_mcp/` is under git):
```bash
git add smol_mcp/requirements.txt && git commit -m "chore(smol-mcp): venv + requirements"
```

---

### Phase 1 — Core: clean tools over stdio MCP

#### Task 1.1: Assemble a clean `tools_lib.py` (no top-level side effects)

**Objective:** Produce `smol_mcp/tools_lib.py` containing *only* the imports,
the chosen tool classes, the helper classes they reference, and the two shared
constants (`BULLETPROOF_FUNCTIONS`, `csv_field_delim`) — with every side-effectful
top-level statement (chrome, network, `matplotlib.use`, `logging.basicConfig`,
folder setup) **excluded**.

**Files:**
- Create: `smol_mcp/assemble_tools_lib.py`
- Create (generated): `smol_mcp/tools_lib.py`

**Approach:** reuse `extract_smaltools.py`'s `ast` parsing. Select by class name
so the script is explicit and reviewable.

**Phase-1 tool set** (self-contained or `safe_dir`-only; the first one, `RVGTool`,
is also the zero-external-dep validation tool):
```python
PHASE1_TOOLS = [
    "RVGTool",
    "LCSAnalyzerTool",
    "SafeOpenTool",
    "CSV2TUPTool",
    "PDF2TXTTool",
    "PDFAnalyzerTool",
    "OCR2TXTTool",
    "PDFSplitterToolGv2",
    "MarkdownToPDFTool",
    "ListBPFunctionsTool",
    "GetBPFunctionCodeTool",
    "SimAnnealTool",
    "SAAnalysisTool",
    "GATool",
    "GAAnalysisTool",
    "BOToolGv1_v3_2_0",
    "BOAnalysisTool",
]
```
**Helper classes** these tools reference (must be in scope):
```python
PHASE1_HELPERS = [
    "LCG", "RandomVariateGenerator",          # RVGTool
    "_TextProcessor",                         # LCSAnalyzerTool
]
```
**Module-level constants** (must be in scope, read via `globals()`):
```python
PHASE1_CONSTS = ["BULLETPROOF_FUNCTIONS", "csv_field_delim"]
```
*(The web-search tools are intentionally **deferred** to Task 1.5 — they pull in
`searxservers`/network setup that is hard to isolate cleanly.)*

**Step 1:** Write `assemble_tools_lib.py`:
```python
#!/usr/bin/env python3
"""Assemble smol_mcp/tools_lib.py from the original monolith (read-only).
Pulls out chosen tool classes, helper classes, and shared constants by
name via ast, and emits a clean module with an explicit import header and
NO top-level side effects.
"""
import ast, os

SRC = r"C:\Users\MATTIA\AppData\Local\hermes\attachments\MyGsearchSmolAgentsZ0WEM_OpAI_SWI_V04_i_SA_s_n_GA_BO_Gv16.py"
OUT = os.path.join(os.path.dirname(__file__), "tools_lib.py")

PHASE1_TOOLS = [ ... ]   # as above
PHASE1_HELPERS = [ ... ] # as above
PHASE1_CONSTS  = [ ... ] # as above

# Imports the chosen tools need at module level in tools_lib.py.
HEADER = '''"""Auto-generated clean tool library. DO NOT EDIT BY HAND.
Generated by assemble_tools_lib.py from MyGsearchSmolAgentsZ0WEM_OpAI_SWI_V04_i_SA_s_n_GA_BO_Gv16.py
(49 smolagents Tool classes + helpers + shared constants, side effects excluded).
"""
import os, re, io, csv, json, math, time, logging
import numpy as np
from collections import deque
from functools import partial
from typing import Dict, List, Type, Optional, Any
from smolagents.tools import Tool
from smolagents.local_python_executor import LocalPythonExecutor, CodeOutput

logger = logging.getLogger("smol_mcp.tools")
'''

with open(SRC, encoding="utf-8") as f:
    text = f.read()
tree = ast.parse(text)

classes = {}
for n in ast.walk(tree):
    if isinstance(n, ast.ClassDef):
        classes[n.name] = n

top_assigns = {}
for n in tree.body:
    if isinstance(n, ast.Assign):
        for t in n.targets:
            if isinstance(t, ast.Name):
                top_assigns[t.id] = n
    elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
        top_assigns[n.target.id] = n

parts = [HEADER]
for name in PHASE1_HELPERS + PHASE1_TOOLS:
    node = classes[name]
    parts.append(ast.get_source_segment(text, node))
    parts.append("\n")
for name in PHASE1_CONSTS:
    node = top_assigns[name]
    parts.append(ast.get_source_segment(text, node))
    parts.append("\n")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(parts))
print(f"Wrote {OUT} ({os.path.getsize(OUT)} bytes): "
      f"{len(PHASE1_TOOLS)} tools, {len(PHASE1_HELPERS)} helpers, {len(PHASE1_CONSTS)} consts")
```

**Step 2:** Run it:
```bash
cd D:/Source/hermes-dir/smol_mcp
.venv/Scripts/python.exe assemble_tools_lib.py
```
Expected: `Wrote .../tools_lib.py (...) : 17 tools, 2 helpers, 2 consts`.

**Step 3:** Verify the module imports **cleanly** (no side effects, no missing names):
```bash
.venv/Scripts/python.exe -c "import tools_lib; print('imported ok'); print('RVG' , tools_lib.RVGTool().name)"
```
Expected: `imported ok` then `RVG random_variate_generator`.
If a `NameError` appears (e.g. a tool references a helper not in
`PHASE1_HELPERS`), add that helper name to `PHASE1_HELPERS` and re-run — repeat
until the import is clean. (This is the main iterative loop of this task.)

**Step 4:** Commit:
```bash
git add smol_mcp/assemble_tools_lib.py smol_mcp/tools_lib.py
git commit -m "feat(smol-mcp): assemble clean tools_lib (Phase 1 tools)"
```

#### Task 1.2: Build `mcp_server.py` with the generic wrapper

**Objective:** A stdio `MCPServer` that wraps each smolagents `Tool` instance as
an MCP tool, deriving the schema from the tool's `forward()` signature (the
mechanism verified in §1.2).

**Files:** Create `smol_mcp/mcp_server.py`

**Complete code:**
```python
#!/usr/bin/env python3
"""smol_mcp — expose smolagents tools to Hermes as a stdio MCP server.

Pattern follows Hermes's own hermes-agent/mcp_serve.py (mcp 2.0.0):
    from mcp.server import MCPServer
    @server.tool()  def f(...) -> str: ...
    await server.run_stdio_async()
"""
import asyncio, inspect, json, logging, sys, types
from typing import Any, Optional

from mcp.server import MCPServer
import tools_lib

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
log = logging.getLogger("smol_mcp")

# ---- safe_dir defaults (mirror original L8563-8575) ----
BASE = r"C:\Users\MATTIA\smol_mcp_data"          # configurable via env
import os
SAFE_DIR  = os.environ.get("SMOL_MCP_SAFE_DIR",  os.path.join(BASE, "uploads"))
PLOTS_DIR = os.environ.get("SMOL_MCP_PLOTS_DIR", os.path.join(BASE, "plots"))
for d in (SAFE_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)

# ---- instantiate the Phase-1 tool graph ONCE (mirror tools_list L8627-8684) ----
def build_tools() -> dict:
    t = {}
    t["random_variate_generator"] = tools_lib.RVGTool()
    t["lcs_analyzer"]             = tools_lib.LCSAnalyzerTool()
    t["safe_open_tool"]           = tools_lib.SafeOpenTool(safe_dir=SAFE_DIR)
    t["csv_2_tup_tool"]           = tools_lib.CSV2TUPTool(safe_dir=SAFE_DIR)
    t["pdf_2_txt_tool"]           = tools_lib.PDF2TXTTool(safe_dir=SAFE_DIR)
    t["pdf_analyzer_tool"]        = tools_lib.PDFAnalyzerTool(safe_dir=SAFE_DIR)
    t["ocr_2_txt_tool"]           = tools_lib.OCR2TXTTool(safe_dir=SAFE_DIR)
    t["pdf_splitter_tool"]        = tools_lib.PDFSplitterToolGv2(safe_dir=SAFE_DIR)
    t["markdown_to_pdf_tool"]     = tools_lib.MarkdownToPDFTool(safe_dir=SAFE_DIR)
    t["list_bulletproof_functions"] = tools_lib.ListBPFunctionsTool()
    t["get_bulletproof_function_code"] = tools_lib.GetBPFunctionCodeTool()
    t["simulated_annealing_optimizer"] = tools_lib.SimAnnealTool(safe_dir=SAFE_DIR)
    t["sa_analysis_tool"]         = tools_lib.SAAnalysisTool(safe_dir=SAFE_DIR)
    t["genetic_algorithm_optimizer"] = tools_lib.GATool(safe_dir=SAFE_DIR)
    t["ga_analysis_tool"]         = tools_lib.GAAnalysisTool(safe_dir=SAFE_DIR)
    t["bayesian_optimizer"]       = tools_lib.BOToolGv1_v3_2_0(safe_dir=SAFE_DIR)
    t["bo_analysis_tool"]         = tools_lib.BOAnalysisTool(safe_dir=SAFE_DIR)
    return t

TOOLS = build_tools()

# ---- generic smolagents-Tool -> MCP-tool wrapper ----
_PYTYPE = {"string": "str", "integer": "int", "number": "float",
           "boolean": "bool", "object": "dict", "array": "list"}

def _forward_params(tool):
    """Return [(name, pytype_str, has_default, default)] for tool.forward()."""
    fwd = tool.forward
    sig = inspect.signature(fwd)
    out = []
    for pname, p in list(sig.parameters.items())[1:]:          # skip 'self'
        ann = getattr(tool, "inputs", {}).get(pname, {})
        smol_type = ann.get("type", "object") if isinstance(ann, dict) else "object"
        pytype = _PYTYPE.get(smol_type, "Any")
        has_def = p.default is not inspect.Parameter.empty
        default = p.default if has_def else None
        # 'any' smol type -> leave untyped (Any) so the schema is permissive
        if smol_type in ("any", ""):
            pytype = "Any"
        out.append((pname, pytype, has_def, default))
    return out

def _make_wrapper(tool: "tools_lib.Tool", name: str):
    """Build a function whose signature mirrors tool.forward(); @server.tool()
    turns its annotations into the MCP input schema and its docstring into the
    tool description. Verified to produce correct schemas + required list."""
    params = _forward_params(tool)
    sig_args, ann, defaults = [], {}, []
    for (pname, pytype, has_def, default) in params:
        sig_args.append(f"{pname}: {pytype}" + (f" = {default!r}" if has_def else ""))
        ann[pname] = pytype
        if has_def:
            defaults.append(default)
    body = f"return _run({name})"
    code = f"def _wrap({', '.join(sig_args)}):\n    {body}\n"
    ns = {}
    exec(code, ns)
    fn = ns["_wrap"]
    fn.__annotations__ = ann
    if defaults:
        fn.__defaults__ = tuple(defaults)
    fn.__name__ = name
    fn.__doc__ = (tool.description or name).strip()
    return fn

def _run(name: str, **kwargs):
    """Dispatch to the tool's forward() and JSON-serialize the result."""
    tool = TOOLS[name]
    try:
        result = tool.forward(**kwargs)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"})
    return _jsonable(result)

def _jsonable(obj):
    """MCP tool results must be JSON. smolagents tools may return dict/str/
    np.ndarray. Convert numpy + non-serializable objects safely."""
    try:
        return json.dumps(obj, default=_np_default)
    except (TypeError, ValueError):
        return json.dumps({"status": "ok", "data": str(obj)})

def _np_default(o):
    if hasattr(o, "tolist"):          # numpy array / scalar
        return o.tolist()
    if isinstance(o, bytes):
        return o.decode("utf-8", "replace")
    return str(o)

def create_server() -> MCPServer:
    server = MCPServer("smol-mcp", instructions=(
        "Reusable smolagents tools: PDF/CSV/text extraction, Markdown->PDF, "
        "random variate generation, LCS text analysis, and black-box optimizers "
        "(simulated annealing, genetic algorithm, Bayesian optimization) with "
        "history-analysis companions."
    ))
    for name, tool in TOOLS.items():
        server.tool()(_make_wrapper(tool, name))
    return server

def main():
    server = create_server()
    async def _run():
        await server.run_stdio_async()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
```

**Step 1:** Write the file.

**Step 2:** Sanity check that the server object builds and lists tools
**without** starting stdio:
```bash
cd D:/Source/hermes-dir/smol_mcp
.venv/Scripts/python.exe -c "import asyncio, mcp_server; s=mcp_server.create_server(); print(len(asyncio.get_event_loop().run_until_complete(s.list_tools())), 'tools')"
```
Expected: a count (≤ 17; fewer if any tool's import/dep failed at build time —
`build_tools()` is called at import, so a missing dep shows here).

**Step 3:** Commit:
```bash
git add smol_mcp/mcp_server.py && git commit -m "feat(smol-mcp): MCPServer + generic tool wrapper"
```

#### Task 1.3: First end-to-end validation with the zero-dep tool (RVG)

**Objective:** Prove the full path (Hermes → stdio MCP → `forward()` → JSON) with
`random_variate_generator`, which needs no external dep beyond smolagents+mcp.

**Files:** Create `smol_mcp/tests/test_mcp_server.py`

**Step 1 — failing test:**
```python
import asyncio, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_rvg_lists_and_calls():
    import mcp_server
    s = mcp_server.create_server()
    async def go():
        tools = await s.list_tools()
        names = {t.name for t in tools}
        assert "random_variate_generator" in names, names
        # call it directly through the same dispatch the MCP layer uses
        out = mcp_server._run("random_variate_generator",
                              distribution="uniform", size=3, seed=42)
        data = json.loads(out)
        assert data["status"] == "success", data
        assert len(data["generated_numbers"]) == 3, data
        return names
    asyncio.run(go())
```

**Step 2:** Run (must fail first — the test file is new, but the assertion is
the real gate; run it to confirm it passes against the real code):
```bash
cd D:/Source/hermes-dir/smol_mcp
.venv/Scripts/python.exe -m pytest tests/test_mcp_server.py -v
```
Expected: `test_rvg_lists_and_calls PASSED`. If it errors, the wrapper or
`build_tools()` is wrong — fix in `mcp_server.py`.

**Step 3 — real stdio handshake (the actual MCP protocol, not just internals):**
```bash
cd D:/Source/hermes-dir/smol_mcp
.venv/Scripts/python.exe - <<'PY'
import asyncio, json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
async def main():
    params = StdioServerParameters(command=".venv/Scripts/python.exe", args=["mcp_server.py"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("LIST:", sorted(t.name for t in tools.tools))
            res = await s.call_tool("random_variate_generator",
                                    {"distribution":"uniform","size":3,"seed":42})
            print("CALL:", res.content[0].text[:160])
asyncio.run(main())
PY
```
Expected: `LIST:` includes `random_variate_generator`; `CALL:` prints a JSON
string with `"status": "success"` and 3 numbers. **This is the key acceptance
proof for Phase 1.**

**Step 4:** Commit:
```bash
git add smol_mcp/tests/test_mcp_server.py && git commit -m "test(smol-mcp): RVG end-to-end via stdio"
```

#### Task 1.4: Register the server in Hermes `config.yaml`

**Objective:** Make the tools appear in Hermes as a stdio MCP server.

**Files:** Modify `C:\Users\MATTIA\AppData\Local\hermes\config.yaml` → `mcp_servers:`

**Step 1:** Read the current `mcp_servers:` block (lines ~642–651) to preserve it.

**Step 2:** Add the stdio entry (use a **forward-slash native path** for the
server script and the venv python; add `env` if `SMOL_MCP_SAFE_DIR` should be
pinned). Per user rule, back up the config first — do **not** clobber it:
```bash
cp C:/Users/MATTIA/AppData/Local/hermes/config.yaml \
   C:/Users/MATTIA/AppData/Local/hermes/backups/config.yaml.bak-smolmcp-$(date +%Y%m%d%H%M%S)
```
Then insert under `mcp_servers:`:
```yaml
mcp_servers:
  obscura-mcp:
    url: http://localhost:8086/mcp
  wolfram:
    url: https://agenttools.wolfram.com/mcp
  mcp-wsEM-1.3:
    url: http://localhost:8000/mcp
    headers:
      Authorization: Bearer lkjdo8...fth1
    timeout: 240
  smol-mcp:
    command: C:/Users/MATTIA/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe
    args:
      - C:/Users/MATTIA/Source/hermes-dir/smol_mcp/mcp_server.py
    env:
      SMOL_MCP_SAFE_DIR: C:/Users/MATTIA/smol_mcp_data/uploads
      SMOL_MCP_PLOTS_DIR: C:/Users/MATTIA/smol_mcp_data/plots
```
> **Note:** the `command` python must be the interpreter that has `mcp` +
> `smolagents` installed. If the dedicated `.venv` is preferred, point `command`
> at `C:/Users/MATTIA/Source/hermes-dir/smol_mcp/.venv/Scripts/python.exe`
> instead. Either works as long as both packages are importable by it.

**Step 3:** Restart the Hermes session (or run the MCP reload) so the server is
spawned and its tools discovered.

**Step 4:** Verify in Hermes: the tools should be listed (e.g. via the tool
discovery / `/tools` surface or by asking the agent to call
`random_variate_generator`). Confirm a real call returns JSON.

**Step 5:** Commit config (if versioned):
```bash
git -C <hermes-config-repo> add config.yaml && git commit -m "feat: register smol-mcp stdio server"
```

#### Task 1.5: Add the web-search tools (deferred — network isolation)

`MyGoogleSearchTool(engine='duckduckgo')` and `SearXNGSearchTool` need the
`searxservers` dict and an external search lib. Because their setup lives in
side-effectful module-level code, handle them explicitly:

**Step 1:** In `assemble_tools_lib.py`, add `MyGoogleSearchTool` and
`SearXNGSearchTool` to `PHASE1_TOOLS`, and add the `searxservers` constant +
`xngsearch`/`gsearch` helper functions to the emitted module (pull the function
source segments too, not just classes). Re-run the assembler; fix the import loop.

**Step 2:** Confirm which search backend each uses (read the `forward()` bodies
in `smolagent_tools_extract/tools/MyGoogleSearchTool.py` and
`SearXNGSearchTool.py`), and add any missing package (e.g. `duckduckgo_search` or
`googlesearch`) to `requirements.txt`; reinstall.

**Step 3:** Add both to `build_tools()` in `mcp_server.py`, extend the test, and
re-run the stdio handshake.

**Step 4:** Commit.

---

### Phase 2 — Guarded / heavier tools (do only if needed)

> These expand reuse but each adds a real dependency or a security surface. Do
> them one at a time, each with its own test.

#### Task 2.1: exec-based optimizers (MP variants) — guard required
`SimAnnealToolMP`, `GAToolMP`, `DEToolMP` run `exec()` on arbitrary
`function_code`. Inside smolagents this is sandboxed by `GuardedLocalExecutor`;
over MCP it is **not**. Mitigations (pick one before exposing):
1. **Restrict input** — accept only *named* BULLETPROOF function names, reject
   raw code (wrap `forward` to validate `function_code` against
   `BULLETPROOF_FUNCTIONS` keys).
2. **Sandbox the server process** — run `mcp_server.py` in a restricted user /
   container / AppContainer so a bad `exec` can't touch the host.
Add the tools to the assembler (`PHASE1_TOOLS` → rename `PHASE2_TOOLS`), add
`mpmath` (already in requirements), wire into `build_tools()`, add a test that a
named function works and (for mitigation 1) raw code is rejected. Commit.

#### Task 2.2: LLM-wrapping tools
`LLM_Tool`, `Summarization_Tool`, `Translation_Tool` need `litellm` and the
`llm_tool_config` dict (model, api_base, api_key). Add `litellm` to
requirements, define `llm_tool_config` in `tools_lib.py` (source from the
original `llm_tool_config`), wire the three tools, test with a trivial
summarize/translate call. **Decision needed:** which model endpoint/key to use
in the MCP context (the original pointed at a local `GB10HOST`).

#### Task 2.3: mem0 / ChromaDB memory tools
`CreateChromaDBTool` … `AnalyzeChromaDBTool` need `mem0` + `mem0_tool_config`.
Add `mem0` (+ `chromadb`) to requirements, define `mem0_tool_config`, wire, test
a create→populate→query round-trip on a temp collection.

#### Task 2.4 (optional, read-only): vault inspector
Expose **only** `ArtifactVaultInspectorToolGv3` (read-only `catalog`/`inspect`)
so Hermes can *see* the HDF5 vault without the write path. Instantiate
`ArtifactVaultToolGv16` + `scout` once in `build_tools()`. Do **not** expose the
write/array-returning DSP chain (`slam_fss`, `pulse_match`, `sync_alignment`,
`audio_handler`, `tfd_*`, `emd`) over MCP — they return numpy arrays meant to
flow through the vault, not across the wire.

---

## 5. Verification checklist (definition of done)

- [ ] `smol_mcp/tools_lib.py` imports cleanly with **no** top-level side effects
      (no browser launch, no network, no `matplotlib.use`).
- [ ] `.venv/Scripts/python.exe -m pytest tests/test_mcp_server.py -v` → all pass.
- [ ] The **stdio handshake** (Task 1.3 Step 3) lists the tools and a real
      `call_tool("random_variate_generator", ...)` returns
      `{"status":"success", ...}`.
- [ ] Hermes `config.yaml` has the `smol-mcp` stdio entry; after a session
      restart the tools appear in Hermes and a live call succeeds.
- [ ] Original `MyGsearchSmolAgents...py` is **unmodified** (read-only).
- [ ] Each enabled tool has at least one passing test.

---

## 6. Risks, tradeoffs, open questions

**Risks**
1. **`exec()` exposure (Phase 2 optimizers).** Highest risk. Mitigate with
   input restriction (named functions only) or a sandboxed server process before
   exposing. Keep Phase 1's standard optimizers' raw-code path disabled/validated.
2. **Shared `globals()` state.** `BULLETPROOF_FUNCTIONS` / `csv_field_delim`
   must exist in `tools_lib.py`. The assembler emits them; if a tool references
   another global, the import loop in Task 1.1 Step 3 will surface it — add it.
3. **Serialization.** Tools returning numpy/arrays need `_jsonable` (provided).
   The DSP/vault chain is deliberately **not** exposed for this reason.
4. **Dependency weight.** `spacy`/`pytesseract`/`pdf2image` are heavy and some
   need native binaries (Tesseract). They only block their specific tools; RVG
   and most PDF/CSV tools do not need them.
5. **Import side effects in the original.** We avoid `import`ing the monolith
   entirely (extraction instead), which is also what your "don't modify saved
   versions" rule requires.

**Tradeoffs**
- **Extraction vs. import:** extraction is more upfront work but is deterministic,
  side-effect-free, and keeps the original pristine. Chosen.
- **stdio vs. HTTP:** stdio is lowest-friction (Hermes spawns it; no port/auth).
  Chosen. (HTTP is available later if the server must be shared.)
- **One server vs. many:** one `smol-mcp` server keeps registration simple;
  tools are namespaced by their smolagents `name`.

**Open questions (answer before Phase 2)**
- Which **model endpoint + key** should the LLM-wrapping tools use in the MCP
  context? (Original used a local `GB10HOST`.)
- For Phase 2 exec optimizers: **input restriction** or **sandboxed process**?
- Should the server's `command` use the dedicated `.venv` or the Hermes venv
  (which already has `mcp` but not `smolagents`)? Recommend the dedicated `.venv`.
- Is `smol_mcp/` (or `D:\Source\hermes-dir`) under git for the commit steps?

---

## 7. Suggested execution order (summary)

1. Task 0.1 — venv + deps.
2. Task 1.1 — assemble `tools_lib.py` (iterate on the clean-import loop).
3. Task 1.2 — `mcp_server.py` + wrapper.
4. Task 1.3 — RVG end-to-end stdio proof.  ← **gate: Phase 1 works**
5. Task 1.4 — register in `config.yaml`, verify in Hermes.
6. Task 1.5 — web search tools.
7. Phase 2 (2.1 → 2.3, 2.4 optional) — only as needed, one tool at a time.
