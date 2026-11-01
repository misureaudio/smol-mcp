import mpmath as mp
from mpmath import mpf


class GeneticAlgorithmMP:
    def __init__(self, objective_function, bounds, pop_size=50, 
                 generations=100, mutation_rate=0.1, dps=50, crossover_rate=0.8):
        mp.mp.dps = int(dps)
        self.func = objective_function
        self.bounds = [[mpf(b[0]), mpf(b[1])] for b in bounds]
        self.dim = len(self.bounds)
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.mutation_rate = mpf(mutation_rate)
        self.crossover_rate = mpf(crossover_rate)  # Ensure this is stored

        self.population = []
        self.fitness = []
        self.best_solution = None
        self.best_value = mp.inf

    def _init_pop(self):
        self.population = []
        for _ in range(self.pop_size):
            ind = []
            for i in range(self.dim):
                low, high = self.bounds[i]
                ind.append(low + (high - low) * mp.rand())
            self.population.append(ind)
        self.fitness = [mp.inf] * self.pop_size

    def _evaluate(self):
        for i in range(self.pop_size):
            val = self.func(self.population[i])
            self.fitness[i] = val
            if val < self.best_value:
                self.best_value = val
                self.best_solution = list(self.population[i])

    def _crossover_and_mutate(self, p1, p2):
        child = []
        alpha = mp.rand()
        for k in range(self.dim):
            # Arithmetic Crossover
            gene = (alpha * p1[k]) + ((1 - alpha) * p2[k])

            # Gaussian Mutation
            if mp.rand() < self.mutation_rate:
                low, high = self.bounds[k]
                domain_span = high - low
                mutation_scale = domain_span * mpf('0.1')

                # Box-Muller
                u1 = mp.rand()
                u2 = mp.rand()
                if u1 == 0:
                    u1 = mpf('1e-50')
                z = mp.sqrt(-2 * mp.log(u1)) * mp.cos(2 * mp.pi * u2)

                gene += z * mutation_scale

                # Clip
                if gene < low:
                    gene = low
                if gene > high:
                    gene = high

            child.append(gene)
        return child

    def optimize(self):
        self._init_pop()
        self._evaluate()
        history = []

        for gen in range(self.generations):
            new_pop = []

            # Elitism
            best_idx = self.fitness.index(min(self.fitness))
            new_pop.append(self.population[best_idx])

            while len(new_pop) < self.pop_size:
                # Tournament
                i1 = int(mp.rand() * self.pop_size)
                p1 = self.population[i1]
                i2 = int(mp.rand() * self.pop_size)
                p2 = self.population[i2]

                child = self._crossover_and_mutate(p1, p2)
                new_pop.append(child)

            self.population = new_pop
            self._evaluate()

            # --- METRICS CALCULATION ---
            # 1. Average Energy (Fitness)
            sum_energy = sum(self.fitness)
            avg_energy = sum_energy / self.pop_size

            # 2. Spatial Diversity (Average Standard Deviation across dimensions)
            # This detects if points are clumped together physically
            variance_sum = mpf(0)
            for d in range(self.dim):
                # Extract all values for dimension d
                dim_values = [ind[d] for ind in self.population]
                dim_mean = sum(dim_values) / self.pop_size
                dim_var = sum((x - dim_mean)**2 for x in dim_values) / self.pop_size
                variance_sum += mp.sqrt(dim_var) # Add standard deviation for this dimension

            diversity = variance_sum / self.dim

            history.append({
                'generation': gen,
                'best_energy': self.best_value,
                'best_solution': list(self.best_solution),
                'avg_energy': avg_energy,
                'diversity': diversity
            })

        return self.best_solution, self.best_value, history