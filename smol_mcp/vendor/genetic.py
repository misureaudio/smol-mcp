import numpy as np

class RealGeneticAlgorithm:
    def __init__(self, objective_function, bounds, 
                 pop_size=50, mutation_rate=0.1, crossover_rate=0.8,
                 elitism_count=2, max_generations=100, 
                 mutation_scale=0.5, seed=None):
        """
        Real-Coded Genetic Algorithm for education/research.
        Tracks diversity and allows transparent population inspection.
        """
        self.func = objective_function
        self.bounds = np.asarray(bounds)
        self.dim = self.bounds.shape[0]
        self.pop_size = int(pop_size)
        self.mutation_rate = float(mutation_rate)
        self.crossover_rate = float(crossover_rate)
        self.elitism = int(elitism_count)
        self.max_generations = int(max_generations)
        self.mutation_scale = float(mutation_scale)
        self._rng = np.random.default_rng(seed)

        # Initialize Population
        # Shape: (pop_size, dim)
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        self.population = self._rng.uniform(low, high, (self.pop_size, self.dim))
        self.fitness = np.zeros(self.pop_size)
        
        # Stats
        self.best_solution = None
        self.best_value = np.inf

    def _evaluate(self):
        for i in range(self.pop_size):
            # Handle 1D/nD unpacking for objective function
            ind = self.population[i]
            val = float(self.func(ind if self.dim > 1 else ind[0]))
            self.fitness[i] = val
            if val < self.best_value:
                self.best_value = val
                self.best_solution = ind.copy()

    def _crossover(self, p1, p2):
        # Arithmetic Crossover
        if self._rng.random() < self.crossover_rate:
            alpha = self._rng.random()
            c1 = alpha * p1 + (1 - alpha) * p2
            c2 = (1 - alpha) * p1 + alpha * p2
            return c1, c2
        return p1.copy(), p2.copy()

    def _mutate(self, ind):
        # Gaussian Mutation
        if self._rng.random() < self.mutation_rate:
            # Scale mutation by the domain size
            ranges = self.bounds[:, 1] - self.bounds[:, 0]
            noise = self._rng.normal(0, self.mutation_scale, size=self.dim) * ranges * 0.1
            ind = ind + noise
            # Clip
            ind = np.clip(ind, self.bounds[:, 0], self.bounds[:, 1])
        return ind

    def optimize(self):
        # Initial Eval
        self._evaluate()
        
        # History Schema:
        # generation, best_energy, diversity, avg_energy
        history = []

        for gen in range(self.max_generations):
            # 1. Calculate Diversity (Std Dev of population positions)
            # A low number means the population has collapsed (converged)
            diversity = np.mean(np.std(self.population, axis=0))
            avg_energy = np.mean(self.fitness)
            
            history.append({
                'generation': gen,
                'best_energy': self.best_value,
                'avg_energy': avg_energy,
                'diversity': diversity,
                'best_solution_x': self.best_solution
            })

            # 2. Selection (Tournament)
            new_pop = np.zeros_like(self.population)
            
            # Elitism: Keep best N
            sorted_indices = np.argsort(self.fitness)
            for i in range(self.elitism):
                new_pop[i] = self.population[sorted_indices[i]]

            # Fill rest
            current_idx = self.elitism
            while current_idx < self.pop_size:
                # Tournament 
                i1, i2 = self._rng.integers(0, self.pop_size, 2)
                p1 = self.population[i1] if self.fitness[i1] < self.fitness[i2] else self.population[i2]
                
                i3, i4 = self._rng.integers(0, self.pop_size, 2)
                p2 = self.population[i3] if self.fitness[i3] < self.fitness[i4] else self.population[i4]

                # Crossover
                c1, c2 = self._crossover(p1, p2)

                # Mutate
                c1 = self._mutate(c1)
                new_pop[current_idx] = c1
                current_idx += 1
                
                if current_idx < self.pop_size:
                    c2 = self._mutate(c2)
                    new_pop[current_idx] = c2
                    current_idx += 1

            self.population = new_pop
            self._evaluate()

        return self.best_solution, self.best_value, history
