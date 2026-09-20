#!/usr/bin/env python3
"""Reproduce EXACTLY how Hermes spawns smol-mcp (command/args/env from
config.yaml) and run the MCP handshake against it.
Run:  .venv/Scripts/python.exe tests/verify_hermes_spawn.py
"""
import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
CFG = r"C:\Users\MATTIA\AppData\Local\hermes\config.yaml"
entry = yaml.safe_load(open(CFG, encoding="utf-8"))["mcp_servers"]["smol-mcp"]
command, args, env = entry["command"], entry["args"], entry.get("env", {})

print("Spawning exactly as Hermes does:")
print("  command:", command)
print("  args:", args)
print("  env:", env)


async def main():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=command, args=args,
        env={**os.environ, **env},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print("INIT ok:", init.server_info.name)
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"LIST ok: {len(names)} tools")
            res = await session.call_tool("list_bulletproof_functions", {"verbose": False})
            print("CALL ok:", res.content[0].text[:60].replace("\n", " "), "...")
            print("HERMES-SPAWN E2E OK")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback; traceback.print_exc()
        print("HERMES-SPAWN E2E FAILED:", type(e).__name__, e)
        sys.exit(1)
