#!/usr/bin/env python3
"""Minimal manual JSON-RPC probe of mcp_server.py over a real subprocess.
Feeds initialize -> initialized -> tools/list and dumps raw stdout/stderr.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
SERVER = os.path.join(HERE, "mcp_server.py")

msgs = [
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}',
    '{"jsonrpc":"2.0","method":"notifications/initialized"}',
    '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}',
]
stdin_data = "\n".join(msgs) + "\n"

p = subprocess.run(
    [PY, SERVER], input=stdin_data, capture_output=True, text=True,
    timeout=90, cwd=HERE,
)
print("=== RAW STDOUT (first 800 chars) ===")
print(repr(p.stdout[:800]))
print("=== STDERR (last 30 lines) ===")
print("\n".join(p.stderr.splitlines()[-30:]))
