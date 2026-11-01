#!/usr/bin/env python3
"""End-to-end test for the Phase-2 MP optimizers.

Spawns smol-mcp EXACTLY as Hermes does (command/args/env from config.yaml),
then:
  1. initialize + list_tools  -> expect 11 tools incl. the 3 MP ones
  2. smoke-run each MP tool with minimal params -> expect success + real dps
Run:  .venv/Scripts/python.exe tests/verify_mp_tools.py
"""
import asyncio, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

CFG = r"C:\Users\MATTIA\AppData\Local\hermes\config.yaml"
entry = yaml.safe_load(open(CFG, encoding="utf-8"))["mcp_servers"]["smol-mcp"]

OBJ = "def objective(x):\n    return sum(mp.mpf(v)**2 for v in list(x))"
BOUNDS = "[(-5, 5), (-5, 5)]"
MP_TOOLS = {
    "simulated_annealing_mp": dict(function_code=OBJ, bounds=BOUNDS,
                                   output_filename="t_sa_mp.csv", dps=20, max_iterations=50),
    "genetic_algorithm_mp":   dict(function_code=OBJ, bounds=BOUNDS,
                                   output_filename="t_ga_mp.csv", dps=20, pop_size=8, generations=4),
    "differential_evolution_mp": dict(function_code=OBJ, bounds=BOUNDS,
                                      output_filename="t_de_mp.csv", dps=20,
                                      num_islands=2, island_pop_size=4, generations=3, migration_interval=2),
}


def check(text: str, want_dps: int):
    is_err = '"status": "error"' in text or '"status":"error"' in text
    has_dps = f"dps={want_dps}" in text or f"Precision: {want_dps}" in text or f"dps {want_dps}" in text
    return is_err, has_dps, text


async def main():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=entry["command"], args=entry["args"],
                                   env={**os.environ, **entry.get("env", {})})
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = sorted(t.name for t in (await s.list_tools()).tools)
            print(f"LIST: {len(tools)} tools")
            mp_present = [n for n in tools if n.endswith(("_mp",))]
            print("MP tools present:", mp_present)
            ok_list = len(tools) == 17 and set(MP_TOOLS) <= set(tools)
            print("LIST OK" if ok_list else "LIST FAIL", "\n")

            all_ok = ok_list
            for name, args in MP_TOOLS.items():
                res = await s.call_tool(name, args)
                text = res.content[0].text
                is_err, has_dps, _ = check(text, args["dps"])
                ok = (not is_err) and has_dps
                all_ok = all_ok and ok
                snippet = text.replace("\\n", " ")[:150]
                print(f"[{ 'OK' if ok else 'FAIL' }] {name}: err={is_err} dps={has_dps}")
                print(f"       {snippet}...")
            print("\nMP E2E:", "ALL OK" if all_ok else "FAILURES")
            sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
