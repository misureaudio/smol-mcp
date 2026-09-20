class DEToolMP(Tool):
    name = "differential_evolution_mp"
    description = "High-precision Differential Evolution with Island Model using pure mpmath."
    inputs = {
        # Required arguments do not need 'nullable'
        "function_code": {
            "type": "string",
            "description": "A string of Python code defining the objective function to be minimized. The function must be named 'objective' and accept a list of numbers."
        },
        "bounds": {
            "type": "object",
            "description": "A list of tuples, where each tuple defines the lower and upper bounds for a dimension. Example: [(-5, 5), (-5, 5)]"
        },
        "output_filename": {
            "type": "string",
            "description": "The base filename for the output CSV and plot files. Must end with .csv."
        },

        # Optional arguments MUST have 'nullable: True'
        "dps": {
            "type": "integer",
            "description": "Decimal Precision Setting (dps) for mpmath arbitrary-precision calculations. Default is 50.",
            "nullable": True
        },
        "num_islands": {
            "type": "integer",
            "description": "The number of separate populations (islands) to evolve. Default is 4.",
            "nullable": True
        },
        "island_pop_size": {
            "type": "integer",
            "description": "The size of the population on each individual island. Default is 40.",
            "nullable": True
        },
        "generations": {
            "type": "integer",
            "description": "The total number of generations to run the evolution for. Default is 200.",
            "nullable": True
        },
        "migration_interval": {
            "type": "integer",
            "description": "The number of generations between migration events, where islands exchange individuals. Default is 20.",
            "nullable": True
        }
    }
    # The tool's forward method returns a formatted string, so we must declare it.
    output_type = "string"

    def __init__(self, safe_dir):
        super().__init__()
        self.safe_dir = safe_dir
        logging.info("DEToolMP initialized. (v0.0.3 W.I.P.)")

    def forward(self, function_code, bounds, output_filename, dps=50,
                num_islands=4, island_pop_size=40, generations=200, migration_interval=20):

        # --- START OF DIAGNOSTIC CODE ---
        import mpmath as mp
        print(f"--- DIAGNOSTIC: mpmath module loaded from: {mp.__file__} ---")

        if not output_filename.lower().endswith('.csv'):
            return "Error: output_filename must end with .csv"

        csv_path = os.path.join(self.safe_dir, os.path.basename(output_filename))
        mp.dps = int(dps)

        # === Function loading (reuse your bulletproof logic) ===
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
        # ... (same as GAToolMP: support standard names + custom code)

        # For brevity: assume you paste the same function resolution block
        # → results in: raw_fn, bounds_list


# === 2. Compile the Function and Prepare High-Precision Bounds ===
        namespace = {'mp': mp, 'np': __import__('numpy')}
        try:
            exec(code_to_compile, namespace)
            raw_fn = namespace['objective']
        except Exception as e:
            return f"Error executing function_code: {e}"

        try:
            # Convert bounds to high-precision mpf objects for the optimizer.
            mpf_bounds_list = [[mp.mpf(b[0]), mp.mpf(b[1])] for b in bounds_list]
        except Exception as e:
            return f"Error converting bounds to high-precision: {e}"

        # === Function and Bounds Loading (The "Bulletproof Logic") ===
        # This block is essential to translate agent inputs to Python objects.

        # We need a namespace to execute the function code in.
        '''
        namespace = {'mp': mp, 'np': __import__('numpy')}

        try:
            # Execute the string of code to define the function.
            exec(function_code, namespace)
            # Extract the actual function object from the namespace.
            # We assume the function is named 'objective'.
            raw_fn = namespace['objective']
        except Exception as e:
            return f"Error executing function_code: {e}"

        try:
            # Convert the standard float bounds into high-precision mpf bounds.
            bounds_list = [[mp.mpf(b[0]), mp.mpf(b[1])] for b in bounds]
        except Exception as e:
            return f"Error converting bounds to high-precision: {e}"
        '''
        # === End of Loading Logic ===

        # === Run Island DE ===
        island_de = IslandModelDEMP(
            objective_function=raw_fn,
            bounds=mpf_bounds_list,
            num_islands=num_islands,
            island_pop_size=island_pop_size,
            generations=generations,
            migration_interval=migration_interval,
            dps=dps
        )
        best_sol, best_val, history = island_de.optimize()

        # === Save CSV ===
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            # header = ['generation', 'global_best_energy'] + [f'x{i}' for i in range(len(bounds_list))]
            header = ['generation', 'global_best_energy'] + [f'x{i}' for i in range(len(bounds))]
            writer.writerow(header)
            for h in history:
                row = [h['generation'], mp.nstr(h['global_best_energy'], mp.dps)]
                row += [mp.nstr(x, mp.dps) for x in h['global_best_solution']]
                writer.writerow(row)

        # === Plot ===
        gens = [h['generation'] for h in history]
        energies = [float(h['global_best_energy']) for h in history]
        plt.figure(figsize=(10, 6))
        plt.plot(gens, energies, 's-', linewidth=2)
        # plt.yscale('log')
        plt.yscale('symlog')
        plt.title(f'Island DE Convergence (dps={dps})')
        plt.grid(True, which='both')
        plot_path = csv_path.replace('.csv', '_convergence.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

        return (
            f"Island Differential Evolution Complete\n"
            f"Precision: {dps} digits | Islands: {num_islands}\n"
            f"Best value: {mp.nstr(best_val, 30)}\n"
            f"Solution: {[mp.nstr(x, 20) for x in best_sol]}\n\n"
            f"History: {os.path.abspath(csv_path)}\n"
            f"Plot: {os.path.abspath(plot_path)}\n"
            f"![DE Convergence]({os.path.abspath(plot_path)})"
        )