class SimAnnealTool(Tool):
    name = "simulated_annealing_optimizer"
    description = """
    Finds the global minimum using Simulated Annealing v2.
    Returns the solution and saves history to a CSV.

    STANDARD FUNCTIONS:
    You can simply pass the name of a standard function in 'function_code' (e.g., "ackley", "rastrigin", "rosenbrock").
    This will use pre-validated code and bounds.

    CUSTOM FUNCTIONS:
    Pass the full Python code definition: "def objective(x): return ...".

    CRITICAL:
    The CSV is saved in a secure area. The Plotter cannot read it directly.
    You must read the data using `csv_2_tup_tool` and INJECT the lists into the plotting code.
    """
    inputs = {
        "function_code": {
            "type": "string",
            "description": "Either the Python code defining 'def objective(x): ...' OR the name of a standard test function (e.g., 'rastrigin')."
        },
        "bounds": {
            "type": "object",
            "description": "List of tuples for min/max per dimension. Optional if using a standard function name."
        },
        "output_filename": {
            "type": "string",
            "description": "Filename for history CSV (e.g., 'run1.csv'). Ends in .csv."
        },
        "max_iterations": {
            "type": "integer",
            "description": "Max iterations. Defaults to 10000.",
            "nullable": True
        },
        "initial_temperature": {
            "type": "number",
            "description": "Starting temp. Defaults to 1000.0.",
            "nullable": True
        },
        "cooling_rate": {
            "type": "number",
            "description": "Cooling rate (0 < r < 1). Defaults to 0.95.",
            "nullable": True
        },
        "neighbor_type": {
            "type": "string",
            "description": "Strategy. Options: 'gaussian', 'uniform', 'jump'. Default: 'gaussian'.",
            "nullable": True
        },
        "schedule_type": {
            "type": "string",
            "description": "Schedule. Options: 'exponential', 'log', 'cauchy', 'adaptive'.",
            "nullable": True
        },
        "restart_stagnation": {
            "type": "integer",
            "description": "If provided, restarts/reheats after N iterations without improvement. (e.g., 500).",
            "nullable": True
        },
        "adapt_neighbor": {
            "type": "boolean",
            "description": "If True (default), automatically adjusts step size. If False, uses fixed scale.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, safe_dir):
        super().__init__()
        self.safe_dir = safe_dir
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("SimAnnealTool initialized (v0.6.0: Standard Library Support).")

    def forward(self, function_code: str, bounds, output_filename: str,
                max_iterations: int = 10000, initial_temperature: float = 1000.0,
                cooling_rate: float = 0.95, neighbor_type: str = "gaussian",
                schedule_type: str = "exponential", restart_stagnation: int = None,
                adapt_neighbor: bool = True):
        try:
            # 0. Secure File Path
            if not output_filename.lower().endswith('.csv'):
                return f"Error: The output_filename '{output_filename}' must end with .csv."

            csv_path = os.path.join(self.safe_dir, os.path.basename(output_filename))

            # ─────────────────────────────────────────────────────────────────
            # 1. FUNCTION SWITCHER LOGIC (New in v0.6.0)
            # ─────────────────────────────────────────────────────────────────
            clean_name = function_code.strip().lower()
            known_global_min = None
            is_predefined = False

            # Check if user passed a standard name
            if clean_name in BULLETPROOF_FUNCTIONS:
                logging.info(f"SimAnnealTool: Using standard function '{clean_name}'")
                data = BULLETPROOF_FUNCTIONS[clean_name]

                # Replace the name with the actual Python code
                function_code = data["code"].strip()
                known_global_min = data.get("global_min")
                is_predefined = True

                # Bounds Logic for Standard Functions:
                # Use User bounds if provided, otherwise fall back to Dictionary bounds
                if not bounds:
                    bounds_list = data["bounds"]
                else:
                    # User wants to override standard bounds
                    if isinstance(bounds, str):
                        try:
                            bounds_list = ast.literal_eval(bounds)
                        except Exception as e:
                            return f"Error parsing user override bounds: {e}"
                    else:
                        bounds_list = bounds
            else:
                # Custom Function Path (Legacy)
                if isinstance(bounds, str):
                    try:
                        bounds_list = ast.literal_eval(bounds)
                    except Exception as e:
                        return f"Error parsing bounds: {e}"
                else:
                    bounds_list = bounds

            # ─────────────────────────────────────────────────────────────────
            # 2. Neighbors & Schedule
            # ─────────────────────────────────────────────────────────────────
            n_map = {"gaussian": sa_neighbors.gaussian_neighbor, "uniform": sa_neighbors.uniform_neighbor, "jump": sa_neighbors.jump_neighbor}
            selected_neighbor = n_map.get(str(neighbor_type).lower(), sa_neighbors.gaussian_neighbor)

            valid_schedules = ["exponential", "log", "cauchy", "adaptive"]
            s_type = str(schedule_type).lower() if str(schedule_type).lower() in valid_schedules else "exponential"

            # 3. Function Compilation
            local_scope = {}
            # Allow common math libraries in exec scope
            global_scope = {'np': np, 'math': math, 'abs': abs, 'min': min, 'max': max, 'pow': pow}
            try:
                exec(function_code, global_scope, local_scope)
            except Exception as e:
                return f"Syntax error in function code: {e}"

            if 'objective' not in local_scope:
                return "Error: Function code must define 'def objective(x):'"

            raw_objective_fn = local_scope['objective']

            def objective_wrapper(x):
                if np.isscalar(x):
                    return raw_objective_fn(np.array([x]))
                return raw_objective_fn(x)

            # 4. Initialize SA
            do_adapt = True if adapt_neighbor is None else adapt_neighbor

            # Check if arguments exist in the loaded class to prevent TypeError if legacy lib is loaded
            init_args = {}
            import inspect
            sig = inspect.signature(SimulatedAnnealing.__init__)
            if 'restart_stagnation' in sig.parameters:
                init_args['restart_stagnation'] = restart_stagnation
                init_args['restart_mode'] = "reheat"
            if 'adapt_neighbor' in sig.parameters:
                init_args['adapt_neighbor'] = do_adapt

            sa = SimulatedAnnealing(
                objective_function=objective_wrapper,
                bounds=bounds_list,
                initial_temperature=initial_temperature,
                cooling_rate=cooling_rate,
                max_iterations=max_iterations,
                neighbor_fn=selected_neighbor,
                schedule_type=s_type,
                **init_args
            )
            # Smart history container: use deque if run will be huge
            use_deque = max_iterations > 50_000
            # deque when max_iterations > 50000
            if use_deque:
                # Keep only last 100k points + all best-improvement points
                history = deque(maxlen=100_000)
            else:
                history = []
            # 5. Optimize
            best_sol, best_val, history = sa.optimize()

            # 6. Write CSV (Robust to version mismatch)
            delimiter_char = globals().get('csv_field_delim', ';')

            # Determine dimensions from first record
            first_record = history[0]
            if len(first_record) >= 1:
                first_pos = first_record[0]
            else:
                return "Error: Optimization returned empty history."

            if np.isscalar(first_pos):
                x_columns = ['x']
            else:
                x_columns = [f'x{i}' for i in range(len(first_pos))]

            fieldnames = ['iteration', 'energy', 'best_so_far', 'accepted', 'temperature', 'neighbor_scale'] + x_columns

            with open(csv_path, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter_char)
                writer.writeheader()

                for it, row_data in enumerate(history):
                    # Robust unpacking (Handle v1 vs v6 tuples)
                    if len(row_data) == 6:
                        pos, energy_val, best_val_hist, accepted_flag, temp, scale = row_data
                    elif len(row_data) == 2:
                        pos, energy_val = row_data
                        best_val_hist, accepted_flag, temp, scale = energy_val, 1, 0.0, 0.0
                    else:
                        continue

                    row = {
                        'iteration': it,
                        'energy': energy_val,
                        'best_so_far': best_val_hist,
                        'accepted': int(accepted_flag),
                        'temperature': temp,
                        'neighbor_scale': scale,
                    }
                    if np.isscalar(pos):
                        row['x'] = float(pos)
                    else:
                        for i, val in enumerate(pos):
                            row[f'x{i}'] = float(val)
                    writer.writerow(row)

            if isinstance(best_sol, np.ndarray):
                sol_list = best_sol.tolist()
            else:
                sol_list = best_sol

            # 7. Construct Result Message
            regret_msg = ""
            if known_global_min is not None:
                regret = abs(best_val - known_global_min)
                regret_msg = f"Known Global Min: {known_global_min}\nRegret (Error): {regret:.6e}\n"

            result_msg = (f"Optimization Successful.\n"
                          f"Function: {clean_name if is_predefined else 'Custom'}\n"
                          f"Global Minimum Found: {best_val}\n"
                          f"{regret_msg}"
                          f"At Coordinates: {sol_list}\n"
                          f"History saved to: {csv_path}\n"
                          f"Settings: Adapt={do_adapt}, Restart={restart_stagnation}\n\n"
                          f"*** PLOTTING INSTRUCTIONS ***\n"
                          f"1. Use `csv_2_tup_tool('{output_filename}')`.\n"
                          f"2. Delimiter is '{delimiter_char}'.\n"
                          f"3. INJECT lists into python_plotter code using f-strings.\n"
                          f"   Example:\n"
                          f"   ```python\n"
                          f"   header, rows = csv_2_tup_tool('{output_filename}')\n"
                          f"   # Col 0=Iter, 1=Energy, 2=Best, 4=Temp\n"
                          f"   x = [float(r[0]) for r in rows]\n"
                          f"   y = [float(r[1]) for r in rows]\n"
                          f"   plot_code = f'''\n"
                          f"   plt.plot({{x}}, {{y}})\n"
                          f"   '''\n"
                          f"   python_plotter(plot_code, 'out.png')\n"
                          f"   ```")

            print(result_msg)
            return result_msg

        except Exception as e:
            logging.exception("Error in SimAnnealTool")
            return f"An error occurred: {str(e)}"