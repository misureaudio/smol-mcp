import numpy as np
# from copy import deepcopy
'''
from .neighbors import gaussian_neighbor
from .schedules import apply_schedule
'''
# BEGIN Gemini 3 robust import
try:
    from .neighbors import gaussian_neighbor
    from .schedules import apply_schedule
except (ImportError, ModuleNotFoundError):
    from neighbors import gaussian_neighbor
    from schedules import apply_schedule
# END Gemini 3 robust import


class SimulatedAnnealing:
    def __init__(self, objective_function, bounds, initial_temperature=1.0,
                 cooling_rate=0.99, max_iterations=10000, min_temperature=1e-9,
                 neighbor_scale=0.1, seed=None, neighbor_fn=None, cooling_schedule=None,
                 schedule_type="exponential", restart_stagnation=None, restart_mode="reheat",
                 adapt_neighbor=True, adapt_window=100, adapt_target_accept=0.44):
        """Simulated Annealing optimizer v6.

        New features in v2-v4:
         - restart_mode: one of 'reheat', 'reset', 'random_jump'
         - adaptive neighbor scaling: adjusts neighbor_scale to target acceptance rate
        """
        self.objective_function = objective_function
        self._rng = np.random.default_rng(seed)
        self.initial_temperature = float(initial_temperature)
        self.cooling_rate = float(cooling_rate)
        self.max_iterations = int(max_iterations)
        self.min_temperature = float(min_temperature)
        self.neighbor_scale = float(neighbor_scale)
        self._user_neighbor_fn = neighbor_fn
        # Use a lambda that captures self.neighbor_scale dynamically
        self.neighbor_fn = neighbor_fn or (lambda x, T, rng: gaussian_neighbor(x, T, rng, scale=self.neighbor_scale))
        self.cooling_schedule = cooling_schedule
        self.schedule_type = schedule_type

        # restart/stagnation
        self.restart_stagnation = restart_stagnation  # int iterations with no improvement to trigger restart
        self.restart_mode = restart_mode  # 'reheat' | 'reset' | 'random_jump'

        # adaptive neighbor scaling
        self.adapt_neighbor = bool(adapt_neighbor)
        self.adapt_window = int(adapt_window)
        self.adapt_target_accept = float(adapt_target_accept)

        bounds_arr = np.asarray(bounds, float)
        if bounds_arr.ndim == 1:
            bounds_arr = bounds_arr.reshape(1, 2)
        if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
            raise ValueError("bounds must be shape (n,2) or [low, high]")
        self.bounds = bounds_arr
        self.dim = bounds_arr.shape[0]

        self.best_solution = None
        self.best_value = np.inf

    def _random_within_bounds(self):
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        return self._rng.uniform(low, high)

    def _generate_candidate(self, current, T):
        # call neighbor_fn (which by default uses current self.neighbor_scale)
        cand = self.neighbor_fn(np.asarray(current, float), T, self._rng)
        # ensure within bounds
        return np.clip(cand, self.bounds[:, 0], self.bounds[:, 1])

    @staticmethod
    def _acceptance_probability(cv, nv, T):
        if nv <= cv:
            return 1.0
        return float(np.exp(-(nv - cv) / max(T, 1e-12)))

    def optimize(self, initial_solution=None, verbose=False):
        if initial_solution is None:
            current = self._random_within_bounds()
        else:
            current = np.asarray(initial_solution, float)
            if current.shape != (self.dim,):
                if self.dim == 1:
                    current = np.array([float(initial_solution)])
                else:
                    raise ValueError("initial_solution dimension mismatch")
        value = float(self.objective_function(current if self.dim > 1 else current[0]))
        T = self.initial_temperature

        self.best_solution = current.copy()
        self.best_value = value

        # --- FIX: Initialize history with FULL schema to match append() structure ---
        # Schema: (pos, energy, best_so_far, accepted, temp, scale)
        detailed_history = [(current.copy(), value, value, True, T, self.neighbor_scale)]

        last_improvement = 0
        accept_count_window = 0
        attempts_window = 0

        for it in range(self.max_iterations):
            if T <= self.min_temperature:
                if verbose:
                    print("Temperature reached minimum.")
                break

            cand = self._generate_candidate(current, T)
            cand_val = float(self.objective_function(cand if self.dim > 1 else cand[0]))
            ap = self._acceptance_probability(value, cand_val, T)
            accepted = False
            if ap > self._rng.random():
                current = cand
                value = cand_val
                accepted = True

            # update acceptance counters for adaptive scaling
            attempts_window += 1
            if accepted:
                accept_count_window += 1

            if value < self.best_value:
                self.best_solution = current.copy()
                self.best_value = value
                last_improvement = 0
            else:
                last_improvement += 1

            # Append consistent history tuple
            detailed_history.append((
                                        current.copy(),           # position
                                        value,                    # current energy
                                        self.best_value,          # best so far
                                        accepted,                 # bool: was this move accepted?
                                        T,                        # current temperature
                                        self.neighbor_scale       # current scale (even if not adapting)
                                    ))

            # adaptive neighbor scaling every adapt_window iterations
            if self.adapt_neighbor and (attempts_window >= self.adapt_window):
                accept_rate = accept_count_window / max(attempts_window, 1)
                # simple proportional control: if accept_rate < target -> increase scale, else decrease
                if accept_rate < self.adapt_target_accept:
                    # increase neighbor scale to explore more
                    self.neighbor_scale *= 1.1
                else:
                    # decrease neighbor scale to fine-tune
                    self.neighbor_scale *= 0.9
                # clamp to sensible bounds relative to domain
                domain_ranges = (self.bounds[:, 1] - self.bounds[:, 0])
                max_scale = float(np.max(domain_ranges))
                self.neighbor_scale = float(np.clip(self.neighbor_scale, 1e-9, max_scale))

                # Update closure if using default
                if self._user_neighbor_fn is None:
                    self.neighbor_fn = (lambda x, T, rng: gaussian_neighbor(x, T, rng, scale=self.neighbor_scale))
                attempts_window = 0
                accept_count_window = 0

            # handle restart/stagnation
            if self.restart_stagnation is not None and last_improvement >= self.restart_stagnation:
                if verbose:
                    print(f"Stagnation at iter {it}, applying restart_mode={self.restart_mode}")
                if self.restart_mode == 'reheat':
                    T = max(T, self.initial_temperature * 0.75)
                elif self.restart_mode == 'reset':
                    # hard reset to random point and temperature
                    current = self._random_within_bounds()
                    value = float(self.objective_function(current if self.dim > 1 else current[0]))
                    T = float(self.initial_temperature)
                elif self.restart_mode == 'random_jump':
                    # perturb current aggressively
                    ranges = self.bounds[:,1] - self.bounds[:,0]
                    jump = self._rng.normal(0, ranges * 0.5, size=self.dim)
                    current = np.clip(current + jump, self.bounds[:, 0], self.bounds[:, 1])
                    value = float(self.objective_function(current if self.dim > 1 else current[0]))
                    # small reheat too
                    T = max(T, self.initial_temperature * 0.5)
                else:
                    # unknown mode -> do mild reheat
                    T = max(T, self.initial_temperature * 0.75)
                last_improvement = 0

            # cool
            T = apply_schedule(T, it, self)

        return (float(self.best_solution[0]) if self.dim == 1 else self.best_solution.copy(),
                self.best_value,
                detailed_history)
