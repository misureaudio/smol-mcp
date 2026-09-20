class BOToolGv1_v3_2_0(Tool):
    name = "bayesian_optimizer"
    description = """
    Finds the global minimum using Bayesian Optimization (Gaussian Processes).
    Best for EXPENSIVE functions where you can only afford a few iterations (e.g. < 50).

    OUTPUT:
    - Returns best solution found.
    - Saves history to CSV.

    CRITICAL:
    - Uses `bayesian-optimization` package (v3.2.0+ compatible).
    - Read CSV using `csv_2_tup_tool` and INJECT data into `python_plotter`.
    """
    inputs = {
        "function_code": {
            "type": "string",
            "description": "Python code 'def objective(x): ...' OR a standard function name (e.g., 'rastrigin', 'schwefel')."
        },
        "bounds": {
            "type": "object",
            "description": "List of tuples e.g. [(-5, 5)]. Optional if using a standard function name."
        },
        "output_filename": {
            "type": "string",
            "description": "Filename for history CSV (e.g., 'bo_run1.csv'). Ends in .csv."
        },
        "n_iter": {
            "type": "integer",
            "description": "Total optimization steps (after init). Default 20.",
            "nullable": True
        },
        "init_points": {
            "type": "integer",
            "description": "Number of random exploration steps before model starts. Default 5.",
            "nullable": True
        },
        "acq_type": {
            "type": "string",
            "description": "Acquisition: 'ucb' (Upper Confidence Bound), 'ei' (Expected Improvement), or 'poi' (Probability Of Improvement). Default 'ucb'.",
            "nullable": True
        },
        "kappa": {
            "type": "number",
            "description": "Exploration weight for UCB. Higher (e.g. 5.0)=Explore; Lower (e.g. 1.0)=Exploit. Default 2.5.",
            "nullable": True
        },
        "xi": {
            "type": "number",
            "description": "Exploration weight for EI. Default 0.0.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, safe_dir):
        super().__init__()
        self.safe_dir = safe_dir
        logging.info("BOTool initialized (v0.6.2 updated strictly for BO v3.2.0 constructor signature).")

    def forward(self, function_code: str, bounds, output_filename: str,
                n_iter: int = 20, init_points: int = 5,
                acq_type: str = 'ucb', kappa: float = 2.5, xi: float = 0.0):

        try:
            # 2. Secure File Path
            if not output_filename.lower().endswith('.csv'):
                return f"Error: The output_filename '{output_filename}' must end with .csv."
            csv_path = os.path.join(self.safe_dir, os.path.basename(output_filename))

            # ─────────────────────────────────────────────────────────────────
            # 3. FUNCTION SWITCHER LOGIC
            # ─────────────────────────────────────────────────────────────────
            bulletproof_dict = globals().get('BULLETPROOF_FUNCTIONS', {})
            clean_name = function_code.strip().lower()

            code_to_compile = ""
            bounds_list = []
            known_global_min = None

            if clean_name in bulletproof_dict:
                logging.info(f"BOTool: Using standard function '{clean_name}'")
                data = bulletproof_dict[clean_name]

                # 1. Get Code & Known Min
                code_to_compile = data["code"].strip()
                known_global_min = data.get("global_min")

                # 2. Handle Bounds
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
            # 4. COMPILATION
            # ─────────────────────────────────────────────────────────────────
            local_scope = {}
            global_scope = {'np': np, 'math': math, 'abs': abs, 'min': min, 'max': max, 'pow': pow}

            try:
                exec(code_to_compile, global_scope, local_scope)
            except Exception as e:
                return f"Syntax error in function code: {e}"

            if 'objective' not in local_scope:
                return "Error: Function 'objective(x)' not found in code."

            raw_objective_fn = local_scope['objective']

            # 5. Wrapper: Minimize <-> Maximize translation
            def bo_wrapper(**kwargs):
                # Ensure correct argument order (x0, x1, x2...)
                sorted_keys = sorted(kwargs.keys())
                x_vec = np.array([kwargs[k] for k in sorted_keys])
                val = raw_objective_fn(x_vec)
                return -float(val)

            # 6. Setup BO Parameters
            pbounds = {f'x{i}': b for i, b in enumerate(bounds_list)}

            n_iter = 20 if n_iter is None else n_iter
            init_points = 5 if init_points is None else init_points
            acq = acq_type.lower() if acq_type else 'ucb'
            k_val = 2.5 if kappa is None else kappa
            xi_val = 0.0 if xi is None else xi

            # Construct the AcquisitionFunction instance
            if acq == 'ei':
                utility = ExpectedImprovement(xi=xi_val)
            elif acq == 'poi':
                utility = ProbabilityOfImprovement(xi=xi_val)
            else:
                utility = UpperConfidenceBound(kappa=k_val)

            # 7. Instantiate Optimizer (In v3.2.0 acquisition_function belongs here)
            optimizer = BayesianOptimization(
                f=bo_wrapper,
                pbounds=pbounds,
                random_state=1,
                verbose=0,
                allow_duplicate_points=True,
                acquisition_function=utility
            )

            # 8. Run Optimization
            logging.info(f"BO Start: {acq}, n_iter={n_iter}, init={init_points}")

            # Maximize method is strictly limited to init_points and n_iter
            optimizer.maximize(
                init_points=init_points,
                n_iter=n_iter
            )

            # 9. Extract Results & Write CSV
            best_res = optimizer.max
            best_val_minimized = -best_res['target']  # Invert back

            best_params = [best_res['params'][f'x{i}'] for i in range(len(bounds_list))]

            delimiter_char = globals().get('csv_field_delim', ';')
            history = optimizer.res
            dim_keys = [f'x{i}' for i in range(len(bounds_list))]
            current_best = np.inf

            with open(csv_path, mode='w', newline='') as f:
                fieldnames = ['iteration', 'energy', 'best_so_far'] + dim_keys
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter_char)
                writer.writeheader()

                for i, res in enumerate(history):
                    energy = -res['target']  # Invert back
                    if energy < current_best:
                        current_best = energy

                    row = {
                        'iteration': i,
                        'energy': energy,
                        'best_so_far': current_best
                    }
                    for k in dim_keys:
                        row[k] = res['params'][k]
                    writer.writerow(row)

            # 10. Reporting
            comparison_msg = ""
            if known_global_min is not None:
                diff = abs(best_val_minimized - known_global_min)
                if diff < 1e-3:
                    comparison_msg = f"\n✅ SUCCESS: Hit Global Minimum (Diff: {diff:.2e})"
                else:
                    comparison_msg = f"\n⚠️ GAP: {diff:.4f} away from Global Minimum ({known_global_min})"

            return (f"Bayesian Optimization Complete.\n"
                    f"Global Minimum: {best_val_minimized}\n"
                    f"At Coordinates: {best_params}"
                    f"{comparison_msg}\n"
                    f"History saved to: {csv_path}\n\n"
                    f"*** PLOTTING INSTRUCTIONS ***\n"
                    f"1. Use `csv_2_tup_tool('{output_filename}')`.\n"
                    f"2. Delimiter is '{delimiter_char}'.\n"
                    f"3. INJECT data into `python_plotter` using f-strings.\n"
                    f"   Example:\n"
                    f"   ```python\n"
                    f"   header, rows = csv_2_tup_tool('{output_filename}')\n"
                    f"   iters = [float(r[0]) for r in rows]\n"
                    f"   energy = [float(r[1]) for r in rows]\n"
                    f"   best = [float(r[2]) for r in rows]\n"
                    f"   plot_code = f'''\n"
                    f"   plt.plot({{iters}}, {{energy}}, 'bo', alpha=0.5, label='Sampled')\n"
                    f"   plt.plot({{iters}}, {{best}}, 'r-', linewidth=2, label='Best So Far')\n"
                    f"   '''\n"
                    f"   python_plotter(plot_code, 'bo_plot.png')\n"
                    f"   ```")

        except Exception as e:
            logging.exception("Error in BOTool")
            return f"An error occurred: {str(e)}"