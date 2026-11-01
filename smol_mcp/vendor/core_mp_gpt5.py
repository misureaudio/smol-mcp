import mpmath as mp
from mpmath import mpf


class SimulatedAnnealingMP:
    def __init__(self,
                 objective_function,
                 bounds,
                 initial_temperature=1.0,
                 cooling_rate=0.99,
                 max_iterations=1000,
                 min_temperature='1e-50',
                 neighbor_scale=0.1,
                 dps=50,
                 schedule_type='log',
                 cooling_schedule=None):
        """
        Arbitrary Precision Simulated Annealing with Advanced Cooling Schedules.
        Uses mpmath for all calculations.
        """
        # Set Global Precision
        mp.mp.dps = int(dps)

        self.func = objective_function
        self.initial_temperature = mpf(initial_temperature)
        self.cooling_rate = mpf(cooling_rate)
        self.max_iterations = int(max_iterations)
        self.min_temperature = mpf(min_temperature)
        self.neighbor_scale = mpf(neighbor_scale)

        # Schedule configuration
        self.schedule_type = schedule_type
        self.cooling_schedule = cooling_schedule

        # Bounds: List of [low, high] converted to mpf
        self.bounds = [[mpf(b[0]), mpf(b[1])] for b in bounds]
        self.dim = len(self.bounds)

        self.best_solution = None
        self.best_value = mp.inf

    def _random_within_bounds(self):
        res = []
        for i in range(self.dim):
            low, high = self.bounds[i]
            # mp.rand() returns float in [0, 1)
            val = low + (high - low) * mp.rand()
            res.append(val)
        return res

    def _generate_candidate(self, current, T):
        cand = []
        for i in range(self.dim):
            # Box-Muller transform for normal distribution in mpmath
            u1, u2 = mp.rand(), mp.rand()
            # Handle tiny probability of log(0)
            if u1 == 0:
                u1 = mpf('1e-50')

            z0 = mp.sqrt(-2 * mp.log(u1)) * mp.cos(2 * mp.pi * u2)

            # Step size scales with sqrt(T)
            step = z0 * self.neighbor_scale * mp.sqrt(max(T, mpf('1e-100')))
            new_val = current[i] + step

            # Clip
            low, high = self.bounds[i]
            if new_val < low:
                new_val = low
            if new_val > high:
                new_val = high
            cand.append(new_val)
        return cand

    def _acceptance_probability(self, cv, nv, T):
        if nv <= cv:
            return mpf(1.0)
        try:
            return mp.exp(-(nv - cv) / max(T, mpf('1e-100')))
        except Exception:
            return mpf(0.0)

    def _apply_schedule(self, T, it):
        """
        MP-Compliant temperature update rule.
        Transplanted and adapted from schedules_gpt5.py.
        """
        # 1. User-provided cooling schedule overrides everything
        if self.cooling_schedule is not None:
            # We assume the user function returns something convertible to mpf
            return mpf(self.cooling_schedule(T, it, self))

        stype = (self.schedule_type or "exponential").lower()
        T0 = self.initial_temperature
        Tmin = self.min_temperature
        alpha = self.cooling_rate

        # Current iteration as mpf
        k = mpf(it) 
        max_iter = mpf(self.max_iterations)

        # ----------------------------------------------------------
        # Standard schedules (MP adapted)
        # ----------------------------------------------------------

        if stype == "exponential":
            # T_k = T * alpha
            return max(T * alpha, Tmin)

        elif stype == "linear":
            # T_k = T0 - k * step
            # Ensure max_iterations is at least 1 to avoid div/0
            denom = max(max_iter, mpf(1))
            step = (T0 - Tmin) / denom
            return max(T0 - step * k, Tmin)

        elif stype == "log":
            # T_k = T0 / log(2 + k)
            return max(T0 / mp.log(2 + k), Tmin)

        elif stype == "fast":
            # T_k = T0 / (1 + beta*k)
            beta = 1 - alpha
            return max(T0 / (1 + beta * k), Tmin)

        elif stype == "cauchy":
            # T_k = T0 / (1 + k)
            return max(T0 / (1 + k), Tmin)

        elif stype == "adaptive":
            # Gentle decay
            decay = mpf('0.995')
            return max(T * decay, Tmin)

        # ----------------------------------------------------------
        # High-quality schedules (MP adapted)
        # ----------------------------------------------------------

        elif stype == "lundy":
            # Lundy–Mees schedule: T_{k+1} = T_k / (1 + β*T_k)
            beta = mpf('0.01') # Small beta for gentle decay
            return max(T / (1 + beta * T), Tmin)

        elif stype == "aarts":
            # Aarts–Korst: T_k = T0 / (1 + log(1+k))
            return max(T0 / (1 + mp.log(1 + k)), Tmin)

        elif stype == "vfa":
            # Very Fast Annealing (Ingber):
            # T_k = T0 * exp( -c * k^(1/d) )
            d = mpf(self.dim)
            c = mpf('1.5')
            # k^(1/d)
            exponent = mp.power(k, 1/d)
            return max(T0 * mp.exp(-c * exponent), Tmin)

        elif stype == "quadratic":
            # T_k = T0 / (1 + a*k^2)
            a = mpf('1e-6')
            return max(T0 / (1 + a * (k ** 2)), Tmin)

        elif stype == "cosine":
            # Cosine annealing
            # T_k = T_min + 0.5*(T0 - T_min)*(1 + cos(pi*k/max_iter))
            half = mpf('0.5')
            term1 = (T0 - Tmin)
            term2 = (1 + mp.cos(mp.pi * k / max_iter))
            return Tmin + half * term1 * term2

        # ----------------------------------------------------------
        # Fallback
        # ----------------------------------------------------------
        return max(T * alpha, Tmin)

    def optimize(self):
        current = self._random_within_bounds()
        value = self.func(current)

        T = self.initial_temperature
        self.best_solution = list(current)
        self.best_value = value

        # History stores raw mpf objects
        history = [(list(current), value, T)]

        for it in range(self.max_iterations):
            if T <= self.min_temperature:
                break

            cand = self._generate_candidate(current, T)
            cand_val = self.func(cand)

            ap = self._acceptance_probability(value, cand_val, T)

            if ap > mp.rand():
                current = cand
                value = cand_val

            if value < self.best_value:
                self.best_solution = list(current)
                self.best_value = value

            # Store history
            history.append((list(current), value, T))

            # Apply MP-compliant schedule
            # 'it + 1' ensures we start calculating schedule for the NEXT step,
            # or treat current 'it' as the step index k.
            T = self._apply_schedule(T, it + 1)

        return self.best_solution, self.best_value, history