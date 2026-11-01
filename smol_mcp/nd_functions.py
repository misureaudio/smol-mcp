"""Vectorized n-D generalizations of bulletproof benchmark functions.

numpy / float64 — visualization grade (plots), not for optimization.
Each entry:
  fn(x: array-like of length n) -> float
  bounds: (lo, hi) applied to every coordinate
  min_x(n) -> list of n floats (known global minimizer)
  min_f(n) -> float (known global minimum value)
Every fn must reproduce the 2-D bulletproof library formula at n=2
(cross-validated by tests/test_bp_plot.py).
"""
import numpy as np


def _rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def _bent_cigar(x):
    x = np.asarray(x, dtype=float)
    return float(x[0] ** 2 + 100.0 * np.sum(x[1:] ** 2))


def _ellipsoid(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    c = 10.0 ** (6.0 * np.arange(n) / (n - 1))
    return float(np.sum(c * x ** 2))


def _rastrigin(x):
    x = np.asarray(x, dtype=float)
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def _griewank(x):
    x = np.asarray(x, dtype=float)
    i = np.arange(1, len(x) + 1)
    return float(np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1.0)


def _ackley(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)


def _schwefel(x):
    # 2-D library form uses 837.9658 = 2 * 418.9829; standard n-D form.
    # Note: 837.9658 is a rounded constant, so the true minimum of this
    # formula is slightly positive (see min_f below).
    return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


# Exact maximizer of h(x) = x*sin(sqrt(x)) on [0, 500] (brentq on h'):
_SCHW_XSTAR = 420.968746359982
_SCHW_HSTAR = 418.98288727243374  # h(_SCHW_XSTAR)


def _zakharov(x):
    x = np.asarray(x, dtype=float)
    # Library 2-D convention: s = 0.5*x1 + 1.0*x2, i.e. coefficients i/2
    # (NOT the sqrt(i) variant found in some literature).
    i = 0.5 * np.arange(1, len(x) + 1)
    s = np.sum(i * x)
    return float(np.sum(x ** 2) + s ** 2 + s ** 4)


def _styblinski_tang(x):
    x = np.asarray(x, dtype=float)
    return float(0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x))


ND_REGISTRY = {
    "rosenbrock": {"fn": _rosenbrock, "bounds": (-5.12, 5.12),
                   "min_x": lambda n: [1.0] * n, "min_f": lambda n: 0.0},
    "bent-cigar": {"fn": _bent_cigar, "bounds": (-2.0, 2.0),
                   "min_x": lambda n: [0.0] * n, "min_f": lambda n: 0.0},
    "ellipsoid": {"fn": _ellipsoid, "bounds": (-1.0, 1.0),
                  "min_x": lambda n: [0.0] * n, "min_f": lambda n: 0.0},
    "rastrigin": {"fn": _rastrigin, "bounds": (-5.12, 5.12),
                  "min_x": lambda n: [0.0] * n, "min_f": lambda n: 0.0},
    "griewank": {"fn": _griewank, "bounds": (-600, 600),
                 "min_x": lambda n: [0.0] * n, "min_f": lambda n: 0.0},
    "ackley": {"fn": _ackley, "bounds": (-32.768, 32.768),
               "min_x": lambda n: [0.0] * n, "min_f": lambda n: 0.0},
    "schwefel": {"fn": _schwefel, "bounds": (-500, 500),
                 "min_x": lambda n: [_SCHW_XSTAR] * n,
                 "min_f": lambda n: n * (418.9829 - _SCHW_HSTAR)},
    "zakharov": {"fn": _zakharov, "bounds": (-5, 5),
                 "min_x": lambda n: [0.0] * n, "min_f": lambda n: 0.0},
    "styblinski-tang": {"fn": _styblinski_tang, "bounds": (-5, 5),
                        "min_x": lambda n: [-2.903463] * n,
                        "min_f": lambda n: -39.1661657 * n},
}
