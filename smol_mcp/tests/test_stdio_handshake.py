#!/usr/bin/env python3
"""End-to-end MCP stdio handshake: spawn mcp_server.py as a subprocess and
drive it with a real MCP client (initialize -> list_tools -> call_tool).
Run:  .venv/Scripts/python.exe tests/test_stdio_handshake.py
"""
import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
SERVER = os.path.join(HERE, "mcp_server.py")


async def main():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=PY, args=[SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print("INIT server:", init.server_info.name)

            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"LIST ({len(names)} tools):", names)

            # 1. zero-dep tool
            res = await session.call_tool(
                "list_bulletproof_functions", {"verbose": False}
            )
            text = res.content[0].text
            print("CALL list_bulletproof_functions:", text[:120].replace("\n", " "), "...")
            assert "ackley" in text and "rastrigin" in text, "BP list missing fns"

            # 2. an optimizer (real engine run). `bounds` is declared
            #    object/required in the tool's `inputs`, but the tool auto-fills
            #    it from the named BULLETPROOF function; the documented way is
            #    to pass an empty list.
            res = await session.call_tool(
                "simulated_annealing_optimizer",
                {"function_code": "ackley", "bounds": [],
                 "output_filename": "e2e_sa.csv", "max_iterations": 30},
            )
            sa_text = res.content[0].text
            print("CALL simulated_annealing_optimizer:", sa_text[:160].replace("\n", " "), "...")
            assert "Error" not in sa_text[:80], f"SA failed: {sa_text[:200]}"

            # 3. a PDF tool with no input file should return a graceful error,
            #    not crash the server (server must stay alive).
            res = await session.call_tool("pdf_2_txt_tool", {"file_name": "nope.pdf"})
            print("CALL pdf_2_txt_tool (missing file):", res.content[0].text[:120].replace("\n", " "))

            # 4. server still alive -> list tools again
            tools2 = await session.list_tools()
            assert len(tools2.tools) == len(names), "server lost tools / died"
            print("\nE2E OK: server stayed alive across calls.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("E2E FAILED:", type(e).__name__, e)
        sys.exit(1)
