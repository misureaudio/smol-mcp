"""bulletproof_extra.py — project-owned additions to BULLETPROOF_FUNCTIONS.

The monolith (attachment) is read-only, so new benchmark entries live here
with the SAME schema as the monolith's BULLETPROOF_FUNCTIONS:

    name -> {"code": <str>, "bounds": <list|str>, "global_min": <float>}

  * base entries : numpy code, bounds as list of [lo, hi]
  * _mp entries  : mpmath code, bounds as a string (MP loader convention)

assemble_tools_lib.py merges BULLETPROOF_EXTRA into the generated
tools_lib.py (idempotent; names already in the monolith copy are skipped).

All global_min values below were measured 2026-10 (coarse grid +
bound-respecting L-BFGS-B / multi-start refinement) — see
smol_mcp_badfuncs/verify_refs.py. None are literature values taken on trust.
"""

BULLETPROOF_EXTRA = {
    # ────────────────────────────────────────────────────────────────────
    # Class 1: ill-conditioned valleys
    # ────────────────────────────────────────────────────────────────────
    "rosenbrock": {
        "code": """
import numpy as np

def objective(x):
    x_val, y_val = x[0], x[1]
    return 100 * (y_val - x_val**2) ** 2 + (1 - x_val) ** 2
""",
        "bounds": [[-5.12, 5.12], [-5.12, 5.12]],
        "global_min": 0.0,
    },
    "rosenbrock_mp": {
        "code": """
# MP version of function: rosenbrock
import mpmath as mp

def objective(x):
    x_val = mp.mpf(x[0])
    y_val = mp.mpf(x[1])
    return 100 * (y_val - x_val**2) ** 2 + (1 - x_val) ** 2
""",
        "bounds": "[[-5.12, 5.12], [-5.12, 5.12]]",
        "global_min": 0.0,
    },
    "matyas": {
        "code": """
import numpy as np

def objective(x):
    x_val, y_val = x[0], x[1]
    return 0.26 * (x_val**2 + y_val**2) - 0.48 * x_val * y_val
""",
        "bounds": [[-10, 10], [-10, 10]],
        "global_min": 0.0,
    },
    "matyas_mp": {
        "code": """
# MP version of function: matyas
import mpmath as mp

def objective(x):
    x_val = mp.mpf(x[0])
    y_val = mp.mpf(x[1])
    return mp.mpf('0.26') * (x_val**2 + y_val**2) - mp.mpf('0.48') * x_val * y_val
""",
        "bounds": "[[-10, 10], [-10, 10]]",
        "global_min": 0.0,
    },
    "bent-cigar": {
        "code": """
import numpy as np

def objective(x):
    x_val, y_val = x[0], x[1]
    return x_val**2 + 100 * y_val**2
""",
        "bounds": [[-2, 2], [-2, 2]],
        "global_min": 0.0,
    },
    "bent-cigar_mp": {
        "code": """
# MP version of function: bent-cigar
import mpmath as mp

def objective(x):
    x_val = mp.mpf(x[0])
    y_val = mp.mpf(x[1])
    return x_val**2 + 100 * y_val**2
""",
        "bounds": "[[-2, 2], [-2, 2]]",
        "global_min": 0.0,
    },
    "ellipsoid": {
        "code": """
import numpy as np

def objective(x):
    x_val, y_val = x[0], x[1]
    return 1.0 * x_val**2 + 1e6 * y_val**2
""",
        "bounds": [[-1, 1], [-1, 1]],
        "global_min": 0.0,
    },
    "ellipsoid_mp": {
        "code": """
# MP version of function: ellipsoid
import mpmath as mp

def objective(x):
    x_val = mp.mpf(x[0])
    y_val = mp.mpf(x[1])
    return x_val**2 + mp.mpf(10**6) * y_val**2
""",
        "bounds": "[[-1, 1], [-1, 1]]",
        "global_min": 0.0,
    },
    # ────────────────────────────────────────────────────────────────────
    # Class 2: roughness knob (classical Holder family; the dict's existing
    # "holder table" is a DIFFERENT custom function that reuses the name)
    # ────────────────────────────────────────────────────────────────────
    "holder n. 10": {
        "code": """
import numpy as np

def objective(x):
    x_val, y_val = x[0], x[1]
    inner = 10 * (x_val * np.sin(x_val) + y_val * np.sin(y_val)) + np.cos(x_val) * np.cos(y_val)
    return -np.abs(inner) ** 0.1
""",
        "bounds": [[-10, 10], [-10, 10]],
        "global_min": -1.6594411760657,
        "global_min_note": "measured 2026-10 (grid + L-BFGS-B); classical Holder T=10 on [-10,10]^2",
    },
    "holder n. 10_mp": {
        "code": """
# MP version of function: holder n. 10
import mpmath as mp

def objective(x):
    x_val = mp.mpf(x[0])
    y_val = mp.mpf(x[1])
    inner = 10 * (x_val * mp.sin(x_val) + y_val * mp.sin(y_val)) + mp.cos(x_val) * mp.cos(y_val)
    return -mp.fabs(inner) ** mp.mpf('0.1')
""",
        "bounds": "[[-10, 10], [-10, 10]]",
        "global_min": -1.6594411760657,
        "global_min_note": "measured 2026-10 (grid + L-BFGS-B); classical Holder T=10 on [-10,10]^2",
    },
    "holder n. 100": {
        "code": """
import numpy as np

def objective(x):
    x_val, y_val = x[0], x[1]
    inner = 10 * (x_val * np.sin(x_val) + y_val * np.sin(y_val)) + np.cos(x_val) * np.cos(y_val)
    return -np.abs(inner) ** 0.01
""",
        "bounds": [[-10, 10], [-10, 10]],
        "global_min": -1.05195263598706,
        "global_min_note": "measured 2026-10 (grid + L-BFGS-B); classical Holder T=100 on [-10,10]^2",
    },
    "holder n. 100_mp": {
        "code": """
# MP version of function: holder n. 100
import mpmath as mp

def objective(x):
    x_val = mp.mpf(x[0])
    y_val = mp.mpf(x[1])
    inner = 10 * (x_val * mp.sin(x_val) + y_val * mp.sin(y_val)) + mp.cos(x_val) * mp.cos(y_val)
    return -mp.fabs(inner) ** mp.mpf('0.01')
""",
        "bounds": "[[-10, 10], [-10, 10]]",
        "global_min": -1.05195263598706,
        "global_min_note": "measured 2026-10 (grid + L-BFGS-B); classical Holder T=100 on [-10,10]^2",
    },
    "easom": {
        "code": """
import numpy as np

def objective(x):
    x_val, y_val = x[0], x[1]
    return -np.cos(x_val) * np.cos(y_val) * np.exp(-((x_val - np.pi) ** 2 + (y_val - np.pi) ** 2))
""",
        "bounds": [[-100, 100], [-100, 100]],
        "global_min": -1.0,
    },
    "easom_mp": {
        "code": """
# MP version of function: easom
import mpmath as mp

def objective(x):
    x_val = mp.mpf(x[0])
    y_val = mp.mpf(x[1])
    return -mp.cos(x_val) * mp.cos(y_val) * mp.exp(-((x_val - mp.pi) ** 2 + (y_val - mp.pi) ** 2))
""",
        "bounds": "[[-100, 100], [-100, 100]]",
        "global_min": -1.0,
    },
    # ────────────────────────────────────────────────────────────────────
    # Class 3: deceptive / equal-depth minima
    # ────────────────────────────────────────────────────────────────────
    "himmelblau": {
        "code": """
import numpy as np

def objective(x):
    x_val, y_val = x[0], x[1]
    return (x_val**2 + y_val - 11) ** 2 + (x_val + y_val**2 - 7) ** 2
""",
        "bounds": [[-5, 5], [-5, 5]],
        "global_min": 0.0,
    },
    "himmelblau_mp": {
        "code": """
# MP version of function: himmelblau
import mpmath as mp

def objective(x):
    x_val = mp.mpf(x[0])
    y_val = mp.mpf(x[1])
    return (x_val**2 + y_val - 11) ** 2 + (x_val + y_val**2 - 7) ** 2
""",
        "bounds": "[[-5, 5], [-5, 5]]",
        "global_min": 0.0,
    },
    "six-hump camel": {
        "code": """
import numpy as np

def objective(x):
    x_val, y_val = x[0], x[1]
    return ((4 - 2.1 * x_val**2 + x_val**4 / 3) * x_val**2
            + x_val * y_val
            + (-4 + 4 * y_val**2) * y_val**2)
""",
        "bounds": [[-3, 3], [-2, 2]],
        "global_min": -1.03162845349,
        "global_min_note": "measured 2026-10 (grid + multi-start)",
    },
    "six-hump camel_mp": {
        "code": """
# MP version of function: six-hump camel
import mpmath as mp

def objective(x):
    x_val = mp.mpf(x[0])
    y_val = mp.mpf(x[1])
    return ((4 - mp.mpf('2.1') * x_val**2 + x_val**4 / 3) * x_val**2
            + x_val * y_val
            + (-4 + 4 * y_val**2) * y_val**2)
""",
        "bounds": "[[-3, 3], [-2, 2]]",
        "global_min": -1.03162845349,
        "global_min_note": "measured 2026-10 (grid + multi-start)",
    },
    "goldstein-price": {
        "code": """
import numpy as np

def objective(x):
    x_val, y_val = x[0], x[1]
    t1 = 1 + (x_val + y_val + 1) ** 2 * (19 - 14 * x_val + 3 * x_val**2 - 14 * y_val + 6 * x_val * y_val + 3 * y_val**2)
    t2 = 30 + (2 * x_val - 3 * y_val) ** 2 * (18 - 32 * x_val + 12 * x_val**2 + 48 * y_val - 36 * x_val * y_val + 27 * y_val**2)
    return t1 * t2
""",
        "bounds": [[-2, 2], [-2, 2]],
        "global_min": 3.0,
    },
    "goldstein-price_mp": {
        "code": """
# MP version of function: goldstein-price
import mpmath as mp

def objective(x):
    x_val = mp.mpf(x[0])
    y_val = mp.mpf(x[1])
    t1 = 1 + (x_val + y_val + 1) ** 2 * (19 - 14 * x_val + 3 * x_val**2 - 14 * y_val + 6 * x_val * y_val + 3 * y_val**2)
    t2 = 30 + (2 * x_val - 3 * y_val) ** 2 * (18 - 32 * x_val + 12 * x_val**2 + 48 * y_val - 36 * x_val * y_val + 27 * y_val**2)
    return t1 * t2
""",
        "bounds": "[[-2, 2], [-2, 2]]",
        "global_min": 3.0,
    },
    "styblinski-tang": {
        "code": """
import numpy as np

def objective(x):
    x_val, y_val = x[0], x[1]
    return 0.5 * (x_val**4 - 16 * x_val**2 + 5 * x_val) + 0.5 * (y_val**4 - 16 * y_val**2 + 5 * y_val)
""",
        "bounds": [[-5, 5], [-5, 5]],
        "global_min": -78.3323314075,
        "global_min_note": "measured 2026-10 (grid + multi-start); standard coeffs 0.5*sum(x^4 - 16x^2 + 5x)",
    },
    "styblinski-tang_mp": {
        "code": """
# MP version of function: styblinski-tang
import mpmath as mp

def objective(x):
    x_val = mp.mpf(x[0])
    y_val = mp.mpf(x[1])
    half = mp.mpf('0.5')
    return half * (x_val**4 - 16 * x_val**2 + 5 * x_val) + half * (y_val**4 - 16 * y_val**2 + 5 * y_val)
""",
        "bounds": "[[-5, 5], [-5, 5]]",
        "global_min": -78.3323314075,
        "global_min_note": "measured 2026-10 (grid + multi-start); standard coeffs 0.5*sum(x^4 - 16x^2 + 5x)",
    },
    "zakharov": {
        "code": """
import numpy as np

def objective(x):
    x_val, y_val = x[0], x[1]
    s = 0.5 * x_val + 1.0 * y_val
    return x_val**2 + y_val**2 + s**2 + s**4
""",
        "bounds": [[-5, 5], [-5, 5]],
        "global_min": 0.0,
    },
    "zakharov_mp": {
        "code": """
# MP version of function: zakharov
import mpmath as mp

def objective(x):
    x_val = mp.mpf(x[0])
    y_val = mp.mpf(x[1])
    s = mp.mpf('0.5') * x_val + y_val
    return x_val**2 + y_val**2 + s**2 + s**4
""",
        "bounds": "[[-5, 5], [-5, 5]]",
        "global_min": 0.0,
    },
    "penalty (kornkorn)": {
        "code": """
import numpy as np

def objective(x):
    x_val, y_val = x[0], x[1]
    return (x_val**2 + y_val**2 - 10) ** 2
""",
        "bounds": [[-4, 4], [-4, 4]],
        "global_min": 0.0,
    },
    "penalty (kornkorn)_mp": {
        "code": """
# MP version of function: penalty (kornkorn)
import mpmath as mp

def objective(x):
    x_val = mp.mpf(x[0])
    y_val = mp.mpf(x[1])
    return (x_val**2 + y_val**2 - 10) ** 2
""",
        "bounds": "[[-4, 4], [-4, 4]]",
        "global_min": 0.0,
    },
    # ────────────────────────────────────────────────────────────────────
    # Class 4: non-smooth / discontinuous
    # ────────────────────────────────────────────────────────────────────
    "plateau": {
        "code": """
import numpy as np

def objective(x):
    x_val, y_val = x[0], x[1]
    return np.round(x_val) ** 2 + np.round(y_val) ** 2
""",
        "bounds": [[-5, 5], [-5, 5]],
        "global_min": 0.0,
    },
    "plateau_mp": {
        "code": """
# MP version of function: plateau
import mpmath as mp

def objective(x):
    x_val = mp.mpf(x[0])
    y_val = mp.mpf(x[1])
    # Python's built-in round() on mpf is banker's rounding, identical to
    # np.round (mpmath exposes no module/context-level round function)
    return round(x_val) ** 2 + round(y_val) ** 2
""",
        "bounds": "[[-5, 5], [-5, 5]]",
        "global_min": 0.0,
    },
    "dejong step": {
        "code": """
import numpy as np

def objective(x):
    x_val, y_val = x[0], x[1]
    return np.floor(np.abs(x_val) + 1) + np.floor(np.abs(y_val) + 1)
""",
        "bounds": [[-5, 5], [-5, 5]],
        "global_min": 2.0,
    },
    "dejong step_mp": {
        "code": """
# MP version of function: dejong step
import mpmath as mp

def objective(x):
    x_val = mp.mpf(x[0])
    y_val = mp.mpf(x[1])
    return mp.floor(mp.fabs(x_val) + 1) + mp.floor(mp.fabs(y_val) + 1)
""",
        "bounds": "[[-5, 5], [-5, 5]]",
        "global_min": 2.0,
    },
}
