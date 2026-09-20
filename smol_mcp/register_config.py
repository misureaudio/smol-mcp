#!/usr/bin/env python3
"""Insert the smol-mcp stdio server entry into Hermes config.yaml mcp_servers.
Idempotent: no-op if 'smol-mcp:' is already present. Keeps the file otherwise
byte-identical (line-based insertion).
"""
import io, sys

CFG = r"C:\Users\MATTIA\AppData\Local\hermes\config.yaml"

with open(CFG, encoding="utf-8") as f:
    lines = f.readlines()

if any(l.strip().startswith("smol-mcp:") for l in lines):
    print("smol-mcp already present; no change.")
    sys.exit(0)

# Find the mcp_servers: line.
mcp_idx = None
for i, l in enumerate(lines):
    if l.rstrip("\n") == "mcp_servers:":
        mcp_idx = i
        break
if mcp_idx is None:
    print("ERROR: mcp_servers: not found"); sys.exit(1)

# Find where the mcp_servers block ends: the next line at column 0 that is a
# top-level key (no leading space) and non-blank.
end_idx = len(lines)
for j in range(mcp_idx + 1, len(lines)):
    l = lines[j]
    if l.strip() == "":
        continue
    if not l[0].isspace():
        end_idx = j
        break

entry = [
    "  smol-mcp:\n",
    "    command: C:/Users/MATTIA/Source/hermes-dir/smol_mcp/.venv/Scripts/python.exe\n",
    "    args:\n",
    "      - C:/Users/MATTIA/Source/hermes-dir/smol_mcp/mcp_server.py\n",
    "    env:\n",
    "      SMOL_MCP_DATA_DIR: C:/Users/MATTIA/smol_mcp_data\n",
    "      SMOL_MCP_REPO_DIR: D:/Source/Repos/MyGSearchSmolagentsEM\n",
]

# Insert at the end of the mcp_servers block (before the next top-level key).
new_lines = lines[:end_idx] + entry + lines[end_idx:]
with open(CFG, "w", encoding="utf-8", newline="") as f:
    f.writelines(new_lines)
print(f"Inserted smol-mcp entry before line {end_idx+1}.")
