class GAToolMP(Tool):
    name = "genetic_algorithm_mp"
    description = """
    High-Precision Genetic Algorithm (Arbitrary Precision).
    Uses 'mpmath' instead of 'numpy'. Best for resolving 'Planck-scale' features.

    CRITICAL:
    1. The 'function_code' MUST be written for mpmath (e.g. `mp.sin`).
    2. Use `get_bulletproof_function_code` or standard names (e.g., 'mogadish_mp').
    3. `dps` sets the decimal places (e.g. 50, 100).
    """
    inputs = {
        "function_code": {
            "type": "string",
            "description": "Python code 'def objective(x): ...' OR standard name."
        },
        "bounds": {
            "type": "object",
            "description": "Bounds e.g. [(-0.1, 0.1)]. Optional if using standard name."
        },
        "output_filename": {
            "type": "string",
            "description": "CSV filename."
        },
        "dps": {
            "type": "integer",
            "description": "Decimal precision. Default 50.",
            "nullable": True
        },
        "pop_size": {
            "type": "integer",
            "description": "Population size.",
            "nullable": True
        },
        "generations": {
            "type": "integer",
            "description": "Max generations.",
            "nullable": True
        },
        "mutation_rate": {
            "type": "number",
            "description": "Prob of mutation.",
            "nullable": True
        },
        "crossover_rate": {
            "type": "number",
            "description": "Prob of crossover.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, safe_dir):
        super().__init__()
        self.safe_dir = safe_dir
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("GAToolMP initialized. (v0.1.0 W.I.P.)")

    def forward(self, function_code, bounds, output_filename, dps=50,
                pop_size=50, generations=100, mutation_rate=0.1, crossover_rate=0.8):

        # 0. Secure Path
        if not output_filename.lower().endswith('.csv'):
            return f"Error: The output_filename '{output_filename}' must end with .csv."
        csv_path = os.path.join(self.safe_dir, os.path.basename(output_filename))

        # 1. Imports
        try:
            from genetic_mp import GeneticAlgorithmMP
        except ImportError:
            return "Error: Could not find 'genetic_mp.py' locally."

        # ─────────────────────────────────────────────────────────────────
        # 2. FUNCTION SWITCHER LOGIC (Enabled & Fixed)
        # ─────────────────────────────────────────────────────────────────
        bulletproof_dict = globals().get('BULLETPROOF_FUNCTIONS', {})
        clean_name = function_code.strip().lower()

        code_to_compile = ""
        bounds_list = []
        known_global_min = None

        if clean_name in bulletproof_dict:
            logging.info(f"GAToolMP: Using standard function '{clean_name}'")
            data = bulletproof_dict[clean_name]
            code_to_compile = data["code"].strip()
            known_global_min = data.get("global_min")

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
            code_to_compile = function_code
            if not bounds:
                return "Error: Bounds required for custom function."
            if isinstance(bounds, str):
                try:
                    bounds_list = ast.literal_eval(bounds)
                except Exception as e:
                    return f"Error parsing bounds: {e}"
            else:
                bounds_list = bounds

        # ─────────────────────────────────────────────────────────────────
        # 3. COMPILATION
        # ─────────────────────────────────────────────────────────────────
        # Set global precision before compilation/execution
        mp.dps = int(dps) if dps else 50

        local_scope = {}
        # Inject 'mp' and 'math' (pointing to mp) into the scope
        global_scope = {'mp': mp, 'math': mp}

        try:
            exec(code_to_compile, global_scope, local_scope)
        except Exception as e:
            return f"Syntax Error: {e}"

        if 'objective' not in local_scope:
            return "Error: The code must define a function named 'objective(x)'."

        raw_fn = local_scope['objective']

        # ─────────────────────────────────────────────────────────────────
        # 4. RUN OPTIMIZATION
        # ─────────────────────────────────────────────────────────────────
        try:
            ga = GeneticAlgorithmMP(
                objective_function=raw_fn,
                bounds=bounds_list,
                pop_size=pop_size or 50,
                generations=generations or 100,
                mutation_rate=mutation_rate or 0.1,
                crossover_rate=crossover_rate or 0.8,
                dps=mp.dps
            )

            best_sol, best_val, history = ga.optimize()
        except Exception as e:
            logging.exception("GAToolMP Runtime Error")
            return f"Error during GA execution: {str(e)}"

        # ─────────────────────────────────────────────────────────────────
        # 5. WRITE CSV (High Precision)
        # ─────────────────────────────────────────────────────────────────
        delim = globals().get('csv_field_delim', ';')

        first_sol = history[0]['best_solution']
        dims = len(first_sol)
        x_cols = [f'x{i}' for i in range(dims)]

        try:
            with open(csv_path, 'w', newline='') as f:
                headers = ['generation', 'best_energy', 'avg_energy', 'diversity'] + x_cols
                writer = csv.DictWriter(f, fieldnames=headers, delimiter=delim)
                writer.writeheader()

                for row_data in history:
                    # Convert ALL metrics to high-precision strings
                    csv_row = {
                        'generation': row_data['generation'],
                        'best_energy': mp.nstr(row_data['best_energy'], n=mp.dps),
                        'avg_energy': mp.nstr(row_data['avg_energy'], n=mp.dps),
                        'diversity': mp.nstr(row_data['diversity'], n=mp.dps)
                    }

                    # Unpack solution
                    sol = row_data['best_solution']
                    for i, val in enumerate(sol):
                        csv_row[f'x{i}'] = mp.nstr(val, n=mp.dps)

                    writer.writerow(csv_row)

            # Format final result
            disp_val = mp.nstr(best_val, n=mp.dps)
            disp_sol = [mp.nstr(x, n=mp.dps) for x in best_sol]

        except Exception as e:
            return f"Error writing CSV: {e}"

        # ─────────────────────────────────────────────────────────────────
        # 6. REPORTING
        # ─────────────────────────────────────────────────────────────────
        comparison_msg = ""
        if known_global_min is not None:
            target = mp.mpf(known_global_min)
            diff = abs(best_val - target)
            if diff < mp.mpf('1e-3'):
                comparison_msg = f"\n✅ SUCCESS: Hit Global Minimum (Diff: {mp.nstr(diff, mp.dps)})"
            else:
                comparison_msg = f"\n⚠️ GAP: {mp.nstr(diff, mp.dps)} away"

        return (f"GA MP Optimization Complete (dps={mp.dps}).\n"
                f"Global Min: {disp_val}\n"
                f"At: {disp_sol}"
                f"{comparison_msg}\n"
                f"History saved to {csv_path}\n"
                f"**Note:** CSV contains high-precision strings.")