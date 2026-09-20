#!/usr/bin/env python3
"""Set the smol-mcp command/args to the known-good D: paths, line-by-line,
then verify both files exist. Idempotent and explicit (no substring replace).
"""
import os, re, sys

CFG = r"C:\Users\MATTIA\AppData\Local\hermes\config.yaml"
CMD = "D:/Source/hermes-dir/smol_mcp/.venv/Scripts/python.exe"
ARG = "D:/Source/hermes-dir/smol_mcp/mcp_server.py"

with open(CFG, encoding="utf-8") as f:
    lines = f.readlines()

# Locate the smol-mcp block.
start = None
for i, l in enumerate(lines):
    if l.strip() == "smol-mcp:":
        start = i
        break
if start is None:
    print("ERROR: smol-mcp: not found"); sys.exit(1)

# Block ends at the next top-level (col-0, non-blank) key.
end = len(lines)
for j in range(start + 1, len(lines)):
    l = lines[j]
    if l.strip() and not l[0].isspace():
        end = j
        break

block = lines[start:end]
out = []
for l in block:
    s = l.strip()
    if s.startswith("command:"):
        out.append(f"    command: {CMD}\n")
    elif s.startswith("- ") and "mcp_server.py" in s:
        out.append(f"      - {ARG}\n")
    else:
        out.append(l)
lines[start:end] = out

with open(CFG, "w", encoding="utf-8", newline="") as f:
    f.writelines(lines)

# Verify.
import yaml
e = yaml.safe_load(open(CFG, encoding="utf-8"))["mcp_servers"]["smol-mcp"]
print("command:", repr(e["command"]), "| exists:", os.path.exists(e["command"]))
print("args   :", repr(e["args"]), "| exists:", os.path.exists(e["args"][0]))
ok = os.path.exists(e["command"]) and os.path.exists(e["args"][0])
print("RESULT:", "OK" if ok else "STILL BROKEN")
sys.exit(0 if ok else 1)
