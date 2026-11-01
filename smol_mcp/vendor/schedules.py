import numpy as np


def exponential(T, cooling_rate, iteration, sa): return T * cooling_rate


def logarithmic(T, cooling_rate, iteration, sa): return sa.initial_temperature / np.log(iteration + np.e)


def cauchy(T, cooling_rate, iteration, sa): return sa.initial_temperature / (1 + iteration)


def adaptive(T, cooling_rate, iteration, sa):
    if iteration % 500 == 0 and iteration > 0:
        return T * 1.1
    return T * cooling_rate


def apply_schedule(T, iteration, sa):
    if sa.cooling_schedule is not None:
        return max(sa.cooling_schedule(T, iteration), sa.min_temperature)
    if sa.schedule_type == "exponential": return exponential(T, sa.cooling_rate, iteration, sa)
    if sa.schedule_type == "log": return logarithmic(T, sa.cooling_rate, iteration, sa)
    if sa.schedule_type == "cauchy": return cauchy(T, sa.cooling_rate, iteration, sa)
    if sa.schedule_type == "adaptive": return adaptive(T, sa.cooling_rate, iteration, sa)
    raise ValueError("Unknown schedule type")
