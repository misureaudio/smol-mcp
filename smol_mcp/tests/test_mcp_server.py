"""Smoke tests for the smol-mcp server.

Run:  .venv/Scripts/python.exe -m pytest tests/test_mcp_server.py -v
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_server_lists_all_phase1_tools():
    import mcp_server

    s = mcp_server.create_server()

    async def go():
        tools = await s.list_tools()
        return {t.name for t in tools}

    names = asyncio.run(go())
    expected = {
        "lcs_analyzer",
        "csv_2_tup_tool",
        "pdf_2_txt_tool",
        "pdf_analyzer_tool",
        "ocr_2_txt_tool",
        "pdf_splitter_tool",
        "list_bulletproof_functions",
        "get_bulletproof_function_code",
        "simulated_annealing_optimizer",
        "sa_analysis_tool",
        "genetic_algorithm_optimizer",
        "ga_analysis_tool",
        "bayesian_optimizer",
        "bo_analysis_tool",
    }
    missing = expected - names
    assert not missing, f"missing tools: {missing}; got: {sorted(names)}"


def test_list_bulletproof_lists_and_calls():
    """Zero-dep end-to-end proof: list the BULLETPROOF function library."""
    import mcp_server

    s = mcp_server.create_server()

    async def go():
        tools = await s.list_tools()
        names = {t.name for t in tools}
        assert "list_bulletproof_functions" in names, names
        return names

    asyncio.run(go())
    out = mcp_server._run("list_bulletproof_functions", verbose=False)
    data = json.loads(out)
    text = data if isinstance(data, str) else json.dumps(data)
    assert "ackley" in text and "rastrigin" in text, text


def test_sim_anneal_named_function():
    """Simulated annealing with a named BULLETPROOF function (no raw exec)."""
    import mcp_server

    out = mcp_server._run(
        "simulated_annealing_optimizer",
        function_code="rastrigin",
        bounds=None,
        output_filename="sa_test.csv",
        max_iterations=50,
    )
    data = json.loads(out)
    # The tool returns a summary string; it must not be an error.
    assert "Error" not in str(data)[:80], data
    csv_path = os.path.join(mcp_server.SAFE_DIR, "sa_test.csv")
    assert os.path.exists(csv_path), "history CSV not written"
