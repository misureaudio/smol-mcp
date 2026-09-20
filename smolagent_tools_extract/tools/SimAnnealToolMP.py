class SimAnnealToolMP(Tool):
    name = "simulated_annealing_mp"
    description = """
    High-Precision Simulated Annealing (Arbitrary Precision).
    Uses 'mpmath' (mp) instead of 'numpy'.

    CRITICAL:
    1. The 'function_code' MUST be written for mpmath (e.g. `mp.sin`, `mp.sqrt`, `mp.mpf`).
    2. Use `get_bulletproof_function_code` to retrieve pre-written MP code (e.g. 'mogadish_mp').
    3. `dps` sets the decimal places (e.g. 50, 100).
    """
    inputs = {
        "function_code": {
            "type": "string",
            "description": "Python code defining 'def objective(x): ...' using mpmath."
        },
        "bounds": {
            "type": "string",
            "description": "List of tuples e.g. '[(-1, 1)]' (passed as string literal)."
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
        "max_iterations": {
            "type": "integer",
            "description": "Default 1000. Keep low for MP speed.",
            "nullable": True
        },
        "initial_temperature": {
            "type": "string",
            "description": "Start temp as string (e.g. '1.0').",
            "nullable": True
        },
        "neighbor_scale": {
            "type": "string",
            "description": "Step size scale as string (e.g. '0.001').",
            "nullable": True
        },
        "schedule_type": {
            "type": "string",
            "description": "Schedule. Options: 'exponential', 'log', 'fast, 'cauchy', 'adaptive', 'lundy', 'aarts', 'vfa', 'quadratic', 'cosine'. Default: 'log'",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, safe_dir):
        super().__init__()
        self.safe_dir = safe_dir
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("SimAnnealToolMP initialized (v0.8.0: core_mp_gpt5.py support).")

    def forward(self,
                function_code: str,
                bounds,
                output_filename,
                schedule_type='log',
                dps=50,
                max_iterations=1000,
                initial_temperature='1.0',
                neighbor_scale='0.1'
                ):
        try:
            # 1. Imports
            try:
                from core_mp_gpt5 import SimulatedAnnealingMP
                import mpmath as mp
            except ImportError:
                return "Error: Could not find 'core_mp.py' locally."

            # 2. Secure Path
            if not output_filename.lower().endswith('.csv'):
                return "Error: .csv required."
            csv_path = os.path.join(self.safe_dir, os.path.basename(output_filename))

            # 3. Setup
            import ast
            bounds_list = ast.literal_eval(bounds)

            # Compile with 'mp' injected
            local_scope = {}
            global_scope = {'mp': mp, 'math': mp}
            try:
                exec(function_code, global_scope, local_scope)
            except Exception as e:
                return f"Syntax Error: {e}"

            objective_fn = local_scope['objective']

            # 4. Run
            sa = SimulatedAnnealingMP(
                objective_fn,
                bounds_list,
                initial_temperature=initial_temperature,
                max_iterations=max_iterations,
                neighbor_scale=neighbor_scale,
                dps=dps
            )

            best_sol, best_val, history = sa.optimize()

            # 5. Write CSV (As Strings)
            delim = globals().get('csv_field_delim', ';')
            import csv

            dims = len(best_sol)
            x_cols = [f'x{i}' for i in range(dims)]

            with open(csv_path, 'w', newline='') as f:
                header = ['iteration', 'energy', 'temperature'] + x_cols
                writer = csv.writer(f, delimiter=delim)
                writer.writerow(header)

                for i, (pos, val, temp) in enumerate(history):
                    # Convert MPF objects to full-precision strings
                    s_val = mp.nstr(val, n=dps)
                    # s_temp = mp.nstr(temp, n=6)  # Temp does(n't) need full dps
                    s_temp = mp.nstr(temp, n=dps)  # Temp does(n't) need full dps
                    s_pos = [mp.nstr(p, n=dps) for p in pos]

                    row = [i, s_val, s_temp] + s_pos
                    writer.writerow(row)

            # 6. Format Result
            dsp_dps = 32
            mindsp = min(dps, dsp_dps)
            s_best_val = mp.nstr(best_val, n=mindsp)  # Truncate for display
            # s_best_pos = [mp.nstr(p, n=10) for p in best_sol]  # Truncate for display
            s_best_pos = [mp.nstr(p, n=mindsp) for p in best_sol]  # Truncate for display

            return (f"MP Optimization Complete (dps={dps}).\n"
                    f"Global Min: {s_best_val}\n"
                    f"At: {s_best_pos}\n"
                    f"History saved to {csv_path}\n"
                    f"**Note:** CSV contains high-precision strings. Parse carefully.")

        except Exception as e:
            logging.exception("SimAnnealToolMP Error")
            return f"MP Error: {str(e)}"