Let me pull the exact diff from the submission package I built, so the instructions are verbatim and reproducible.I have the exact diff. Let me confirm the config side still persists (so I state it accurately) and save a durable copy of the patch next to your notes, since the scratch dir gets pruned after 24 h.Config confirmed intact (`timeouts.async_bridge: 14400` + `timeouts.tools.*: 14400`, lines 652–656) — and I saved durable patch copies next to your notes (the scratch dir prunes after 24 h):
- `smol-mcp-image-analysis-timeout/model_tools_async_bridge_timeout.patch` (code only)
- `smol-mcp-image-analysis-timeout/submit_with_tests.patch` (code + tests)

## Reproducing the patch after an update

**What an update wipes vs. keeps:**
- `config.yaml` → the `timeouts.*` keys **persist** (updates don't touch config). Nothing to re-set.
- `model_tools.py` → the code patch **gets wiped** (the file is replaced). This is the only thing to re-apply.

### Step 1 — Re-apply the code patch

**Option A — one-liner** (from the install dir):
```bat
cd "C:\Users\MATTIA\AppData\Local\hermes\hermes-agent"
git apply "C:\Users\MATTIA\source\hermes-dir\smol-mcp-image-analysis-timeout\model_tools_async_bridge_timeout.patch"
```

**Option B — manual edit** (two changes; `Optional` is already imported at the top of the file):

*Change 1* — insert this helper immediately **before** `def _run_async(coro):` (i.e. right after `_get_worker_loop()`):
```python
def _async_bridge_timeout_s() -> Optional[float]:
    """Wall-clock cap (s) for bridging an async tool handler into a worker thread.

    ``timeouts.async_bridge`` (config.yaml) > ``HERMES_ASYNC_TOOL_TIMEOUT_S`` (env) >
    300 s default; ``0``/negative = no cap (``future.result`` waits indefinitely).
    The previous hardcoded 300 s killed legitimately slow tools - e.g. vision on a
    local model that needs minutes per image - with a bare ``TimeoutError`` before the
    underlying HTTP call (which has its own, much larger, configurable timeout) finished.
    """
    from agent.deadline import resolve_timeout
    return resolve_timeout("async_bridge", default=300.0, env_var="HERMES_ASYNC_TOOL_TIMEOUT_S")
```

*Change 2* — inside `_run_async`, in the running-loop branch, replace the one line:
```python
            return future.result(timeout=300)
```
with:
```python
            return future.result(timeout=_async_bridge_timeout_s())
```

### Step 2 — Restart the gateway
The code change needs a fresh process (a config-only change wouldn't — this one does). Same "restart gateway" you just did.

### Step 3 — Verify
From the install dir, with the Hermes Python:
```bat
python -c "import model_tools; print('cap (s):', model_tools._async_bridge_timeout_s())"
```
→ should print `cap (s): 14400.0`.

### The tell that it got wiped again
A vision call dying at **exactly ~300.0 s** with a **bare empty `TimeoutError:`** (no message) in the log = the hardcoded cap firing. That's when you re-run Step 1–3.

**Notes:**
- Precedence is `timeouts.async_bridge` (config) > `HERMES_ASYNC_TOOL_TIMEOUT_S` (env) > 300 s default; `0`/negative = no cap. So the env var is a config-free alternative, but it's only read *by* the patched code — without the patch, `future.result(timeout=300)` is hardcoded and ignores the env var entirely.
- If the config keys ever do vanish, restore with: `hermes config set --force timeouts.async_bridge 14400` (the `--force` is needed for a new key under a known section).