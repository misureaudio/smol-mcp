#!/usr/bin/env python3
"""smol_mcp — expose smolagents tools to Hermes as a stdio MCP server.

Pattern follows Hermes's own hermes-agent/mcp_serve.py (mcp 2.0.0):
    from mcp.server import MCPServer
    @server.tool()  def f(...) -> str: ...
    await server.run_stdio_async()
"""
from __future__ import annotations
import asyncio, contextlib, inspect, json, logging, os, sys
from typing import Any, Optional

# Import the heavy libraries with stdout redirected to stderr. They emit
# deprecation warnings to *stdout* (notably `fitz`), which would corrupt the
# MCP stdio JSON-RPC channel — stdout must carry only JSON-RPC frames. The
# mcp 2.0 stdio transport re-claims fd 1 at startup, so restoring sys.stdout
# to its original object afterwards is safe.
_REAL_STDOUT = sys.stdout
with contextlib.redirect_stdout(sys.stderr):
    from mcp.server import MCPServer
    import tools_lib
sys.stdout = _REAL_STDOUT

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
log = logging.getLogger("smol_mcp")

# ---- optimizer engine modules --------------------------------------------
# The SA / GA / BO optimizer tools reference their engines as module-level
# globals (SimulatedAnnealing, sa_neighbors, RealGeneticAlgorithm,
# BayesianOptimization + acquisition classes). In the original monolith those
# were imported by top-level `try:` blocks; here we load them at startup from
# the original project's local engine modules (SA, GA) and the bayes_opt
# package (BO), then bind them into tools_lib's namespace so the tool methods
# resolve them.
REPO_DIR = os.environ.get(
    "SMOL_MCP_REPO_DIR", r"D:\Source\Repos\MyGSearchSmolagentsEM"
)

def _load_engines() -> None:
    import tools_lib as tl
    if os.path.isdir(REPO_DIR) and REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    # Simulated Annealing engine (local core.py + neighbors.py)
    from core import SimulatedAnnealing
    import neighbors as sa_neighbors
    tl.SimulatedAnnealing = SimulatedAnnealing
    tl.sa_neighbors = sa_neighbors
    # Genetic Algorithm engine (local genetic.py)
    from genetic import RealGeneticAlgorithm
    tl.RealGeneticAlgorithm = RealGeneticAlgorithm
    # Bayesian Optimization engine (bayes_opt package)
    from bayes_opt import BayesianOptimization
    from bayes_opt.acquisition import (
        UpperConfidenceBound, ExpectedImprovement, ProbabilityOfImprovement,
    )
    tl.BayesianOptimization = BayesianOptimization
    tl.UpperConfidenceBound = UpperConfidenceBound
    tl.ExpectedImprovement = ExpectedImprovement
    tl.ProbabilityOfImprovement = ProbabilityOfImprovement
    log.info("optimizer engines loaded (repo=%s)", REPO_DIR)

_load_engines()

# ---- safe_dir defaults (mirror original L8563-8575), overridable via env ----
BASE = os.environ.get("SMOL_MCP_DATA_DIR", os.path.join(os.path.expanduser("~"), "smol_mcp_data"))
SAFE_DIR  = os.environ.get("SMOL_MCP_SAFE_DIR",  os.path.join(BASE, "uploads"))
PLOTS_DIR = os.environ.get("SMOL_MCP_PLOTS_DIR", os.path.join(BASE, "plots"))
for d in (SAFE_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)

# ---- instantiate the Phase-1 tool graph ONCE (mirror tools_list L8627-8684) ----
def build_tools() -> dict:
    t: dict = {}
    t["lcs_analyzer"]              = tools_lib.LCSAnalyzerTool()
    t["csv_2_tup_tool"]            = tools_lib.CSV2TUPTool(safe_dir=SAFE_DIR)
    t["pdf_2_txt_tool"]            = tools_lib.PDF2TXTTool(safe_dir=SAFE_DIR)
    t["pdf_analyzer_tool"]         = tools_lib.PDFAnalyzerTool(safe_dir=SAFE_DIR)
    t["ocr_2_txt_tool"]            = tools_lib.OCR2TXTTool(safe_dir=SAFE_DIR)
    t["pdf_splitter_tool"]         = tools_lib.PDFSplitterToolGv2(safe_dir=SAFE_DIR)
    t["list_bulletproof_functions"]   = tools_lib.ListBPFunctionsTool()
    t["get_bulletproof_function_code"]= tools_lib.GetBPFunctionCodeTool()
    t["simulated_annealing_optimizer"]= tools_lib.SimAnnealTool(safe_dir=SAFE_DIR)
    t["sa_analysis_tool"]          = tools_lib.SAAnalysisTool(safe_dir=SAFE_DIR)
    t["genetic_algorithm_optimizer"]= tools_lib.GATool(safe_dir=SAFE_DIR)
    t["ga_analysis_tool"]          = tools_lib.GAAnalysisTool(safe_dir=SAFE_DIR)
    t["bayesian_optimizer"]        = tools_lib.BOToolGv1_v3_2_0(safe_dir=SAFE_DIR)
    t["bo_analysis_tool"]          = tools_lib.BOAnalysisTool(safe_dir=SAFE_DIR)
    return t

TOOLS = build_tools()

# ---- generic smolagents-Tool -> MCP-tool wrapper ----
# Map smolagents `inputs` type tokens to Python annotation strings.
_PYTYPE = {
    "string": "str", "integer": "int", "number": "float",
    "boolean": "bool", "bool": "bool", "object": "dict",
    "array": "list", "any": "Any", "": "Any",
}

def _forward_params(tool):
    """Return [(name, pytype_str, has_default, default)] for tool.forward().

    A param with no default is treated as optional (default None, nullable
    type) when the tool's own source shows it accepts a falsy/None value
    (e.g. `if not bounds:` or `if x is None`). This corrects smolagents
    `inputs` metadata that over-declares required params — e.g. the optimizers
    declare `bounds` required but auto-fill it from a named function.
    """
    sig = inspect.signature(tool.forward)
    src = ""
    try:
        src = inspect.getsource(tool.forward)
    except (OSError, TypeError):
        pass
    out = []
    inputs = getattr(tool, "inputs", {}) or {}
    # tool.forward is a *bound* method, so its signature has no `self`. Skip a
    # param only if it is actually named `self` (defensive; covers unbound too).
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        ann = inputs.get(pname, {})
        smol_type = ann.get("type", "object") if isinstance(ann, dict) else "object"
        pytype = _PYTYPE.get(smol_type, "Any")
        has_def = p.default is not inspect.Parameter.empty
        default = p.default if has_def else None
        # Some params are polymorphic (the tool branches on isinstance(param,
        # str/list/...)) even though `inputs` declares them a single type.
        # Type those as Any so the schema doesn't reject valid values.
        polymorphic = (
            f"isinstance({pname}, str)" in src
            or f"isinstance({pname}, list)" in src
        )
        if polymorphic:
            pytype = "Any"
        # A param with no default is effectively optional when the source
        # guards it (e.g. `if not bounds:` / `if x is None`). This corrects
        # smolagents `inputs` that over-declare required params — e.g. the
        # optimizers declare `bounds` required but auto-fill it from a named fn.
        effectively_optional = (not has_def) and (
            f"if not {pname}" in src or f"{pname} is None" in src
        )
        if effectively_optional:
            pytype = f"Optional[{pytype}]" if pytype != "Any" else "Any"
            has_def = True
            default = None
        out.append((pname, pytype, has_def, default))
    return out

def _make_wrapper(tool, name: str):
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
    # Build a wrapper whose body collects its named params into a dict and
    # dispatches to _run(name, **dict). Signature/annotations drive the schema.
    code = (
        f"def _wrap({', '.join(sig_args)}):\n"
        f"    _a = {{}}\n"
    )
    for (pname, pytype, has_def, default) in params:
        code += f"    _a[{pname!r}] = {pname}\n"
    code += f"    return _run({name!r}, **_a)\n"
    # Exec into the module globals so annotation names (Any, Optional) and the
    # _run dispatch target all resolve when @server.tool() evaluates the hints.
    exec(code, globals())
    fn = globals()["_wrap"]
    fn.__annotations__ = ann
    if defaults:
        fn.__defaults__ = tuple(defaults)
    fn.__name__ = name
    fn.__doc__ = (tool.description or name).strip()
    return fn

def _run(name: str, **kwargs):
    """Dispatch to the tool's forward() and JSON-serialize the result.

    stdout is redirected to stderr for the duration of the call: some tools
    print() to stdout (e.g. SimAnnealTool, read_dynamic_csv), which would
    otherwise corrupt the MCP JSON-RPC channel. Their stdout goes to the log.
    """
    tool = TOOLS[name]
    try:
        with contextlib.redirect_stdout(sys.stderr):
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
