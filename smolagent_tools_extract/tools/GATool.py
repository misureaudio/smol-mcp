class GATool(Tool):
    name = "genetic_algorithm_optimizer"
    description = """
    Finds the global minimum using a Real-Coded Genetic Algorithm.
    Useful for multimodal functions where Simulated Annealing might get stuck.

    OUTPUT:
    - Returns best solution.
    - Saves history to CSV (Generation, Best Energy, Diversity).

    CRITICAL:
    Read CSV using `csv_2_tup_tool` and INJECT data into `python_plotter`.
    """
    inputs = {
        "function_code": {
            "type": "string",
            "description": "Python code 'def objective(x): ...' OR a standard function name (e.g., 'rastrigin', 'schwefel')."
        },
        "bounds": {
            "type": "object",
            "description": "Bounds e.g. [(-5, 5)]. Optional if using a standard function name."
        },
        "output_filename": {
            "type": "string",
            "description": "CSV filename."
        },
        "pop_size": {"type": "integer", "description": "Population size (default 50).", "nullable": True},
        "generations": {"type": "integer", "description": "Max generations (default 100).", "nullable": True},
        "mutation_rate": {"type": "number", "description": "Prob of mutation (0.0-1.0).", "nullable": True},
        "crossover_rate": {"type": "number", "description": "Prob of crossover (0.0-1.0).", "nullable": True}
    }
    output_type = "string"

    def __init__(self, safe_dir):
        super().__init__()
        self.safe_dir = safe_dir
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("GATool initialized (v0.1.0 with Dictionary Lookup).")

    def forward(self, function_code, bounds, output_filename,
                pop_size=50, generations=100, mutation_rate=0.1,
                crossover_rate=0.8):

        # 0. Secure Path
        if not output_filename.lower().endswith('.csv'):
            return f"Error: The output_filename '{output_filename}' must end with .csv."
        csv_path = os.path.join(self.safe_dir, os.path.basename(output_filename))

        # ─────────────────────────────────────────────────────────────────
        # 1. FUNCTION SWITCHER LOGIC
        # ─────────────────────────────────────────────────────────────────
        # Retrieve the dictionary from the global scope (defined in main script)
        bulletproof_dict = globals().get('BULLETPROOF_FUNCTIONS', {})

        clean_name = function_code.strip().lower()
        known_global_min = None
        code_to_compile = ""
        bounds_list = []

        if clean_name in bulletproof_dict:
            logging.info(f"GATool: Using standard function '{clean_name}'")
            data = bulletproof_dict[clean_name]

            # 1. Get Code
            code_to_compile = data["code"].strip()
            # 2. Get Known Min (for reporting)
            known_global_min = data.get("global_min")

            # 3. Handle Bounds (User Override vs Default)
            if not bounds:
                bounds_list = data["bounds"]
            else:
                if isinstance(bounds, str):
                    try:
                        bounds_list = ast.literal_eval(bounds)
                    except Exception as e:
                        return f"Error parsing user override bounds: {e}"
                else:
                    bounds_list = bounds
        else:
            # Custom Code Mode
            code_to_compile = function_code
            if not bounds:
                return "Error: You must provide 'bounds' for custom functions."
            if isinstance(bounds, str):
                try:
                    bounds_list = ast.literal_eval(bounds)
                except Exception as e:
                    return f"Error parsing bounds: {e}"
            else:
                bounds_list = bounds

        # ─────────────────────────────────────────────────────────────────
        # 2. COMPILATION (Turn String into Callable Function)
        # ─────────────────────────────────────────────────────────────────
        local_scope = {}
        global_scope = {'np': np, 'math': math}
        try:
            exec(code_to_compile, global_scope, local_scope)
        except Exception as e:
            return f"Syntax Error in function code: {e}"

        if 'objective' not in local_scope:
            return "Error: The code must define a function named 'objective(x)'."

        raw_fn = local_scope['objective']

        # Wrapper to handle 1D scalars vs Arrays
        def wrapper(x):
            if np.isscalar(x):
                return raw_fn(np.array([x]))
            return raw_fn(x)

        # ─────────────────────────────────────────────────────────────────
        # 3. RUN OPTIMIZATION
        # ─────────────────────────────────────────────────────────────────
        try:
            ga = RealGeneticAlgorithm(
                objective_function=wrapper,  # Pass the compiled function object!
                bounds=bounds_list,
                pop_size=pop_size or 50,
                max_generations=generations or 100,
                mutation_rate=mutation_rate or 0.1,
                crossover_rate=crossover_rate or 0.8
            )

            best_sol, best_val, history = ga.optimize()
        except Exception as e:
            logging.exception("GATool Runtime Error")
            return f"Error during GA execution: {str(e)}"

        # ─────────────────────────────────────────────────────────────────
        # 4. WRITE CSV
        # ─────────────────────────────────────────────────────────────────
        delim = globals().get('csv_field_delim', ';')

        # Determine dimension for headers based on the first solution in history
        first_sol = history[0]['best_solution_x']
        dims = 1 if np.isscalar(first_sol) else len(first_sol)
        x_cols = [f'x{i}' for i in range(dims)]

        try:
            with open(csv_path, 'w', newline='') as f:
                headers = ['generation', 'best_energy', 'avg_energy', 'diversity'] + x_cols
                writer = csv.DictWriter(f, fieldnames=headers, delimiter=delim)
                writer.writeheader()

                for row in history:
                    csv_row = {
                        'generation': row['generation'],
                        'best_energy': row['best_energy'],
                        'avg_energy': row['avg_energy'],
                        'diversity': row['diversity']
                    }
                    # Unpack solution
                    sol = row['best_solution_x']
                    if np.isscalar(sol):
                        csv_row['x0'] = float(sol)
                    else:
                        for i, v in enumerate(sol):
                            csv_row[f'x{i}'] = float(v)
                    writer.writerow(csv_row)
        except Exception as e:
            return f"Error writing CSV: {e}"

        if isinstance(best_sol, np.ndarray):
            best_sol = best_sol.tolist()

        # ─────────────────────────────────────────────────────────────────
        # 5. REPORTING
        # ─────────────────────────────────────────────────────────────────
        # Compare with known global minimum if available
        comparison_msg = ""
        if known_global_min is not None:
            diff = abs(best_val - known_global_min)
            if diff < 1e-4:
                comparison_msg = f"\n✅ SUCCESS: Hit Global Minimum (Diff: {diff:.2e})"
            else:
                comparison_msg = f"\n⚠️ GAP: {diff:.4f} away from Global Minimum ({known_global_min})"

        return (f"GA Optimization Complete.\n"
                f"Global Min Found: {best_val}\n"
                f"At Coordinates: {best_sol}"
                f"{comparison_msg}\n"
                f"History saved to {csv_path}\n\n"
                f"**PLOT INSTRUCTIONS**:\n"
                f"Plot 'best_energy' vs 'generation' for convergence.\n"
                f"Plot 'diversity' vs 'generation' to see if population collapsed.")