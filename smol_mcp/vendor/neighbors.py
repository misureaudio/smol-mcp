import numpy as np


def gaussian_neighbor(x, T, rng, scale=1.0):
    """Gaussian neighbor that respects a provided scale parameter.

    scale is interpreted as a typical step size (in the domain units).
    We multiply by sqrt(T) to make steps temperature-dependent.
    """
    x = np.asarray(x, float)
    return x + rng.normal(0.0, scale * np.sqrt(max(T, 1e-12)), size=len(x))


def uniform_neighbor(x, T, rng, scale=1.0):
    x = np.asarray(x, float)
    return x + rng.uniform(-scale, scale, size=len(x))


def jump_neighbor(x, T, rng):
    x = np.asarray(x, float)
    if rng.random() < 0.05:
        return x + rng.normal(0, 5.0 * np.sqrt(max(T, 1e-12)), size=len(x))
    return x + rng.normal(0, 1.0 * np.sqrt(max(T, 1e-12)), size=len(x))
