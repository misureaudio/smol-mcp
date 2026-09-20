class BOAnalysisTool(Tool):
    name = "bo_analysis_tool"
    description = """
    Analyzes 'optimization_history.csv' from Bayesian Optimization.
    Diagnoses Exploration (Variance) vs Exploitation (Convergence).
    """
    inputs = {
        "csv_filename": {
            "type": "string",
            "description": "Name of the history CSV file."
        }
    }
    output_type = "string"

    def __init__(self, safe_dir):
        super().__init__()
        self.safe_dir = safe_dir
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("bo_analysis_tool initialized.")

    def forward(self, csv_filename: str):
        if not csv_filename.lower().endswith('.csv'):
            return "Error: .csv required"
        path = os.path.join(self.safe_dir, os.path.basename(csv_filename))
        if not os.path.exists(path):
            return "Error: File not found"

        try:
            delim = globals().get('csv_field_delim', ';')
            with open(path, 'r') as f:
                reader = csv.DictReader(f, delimiter=delim)
                rows = list(reader)

            if not rows:
                return "Empty file."

            energies = np.array([float(r['energy']) for r in rows])
            n = len(energies)

            # Extract positions
            x_keys = [k for k in rows[0].keys() if k.startswith('x')]
            positions = []
            for r in rows:
                positions.append([float(r[k]) for k in x_keys])
            positions = np.array(positions)

            best_idx = np.argmin(energies)
            best_val = energies[best_idx]

            # Metric 1: Exploration Radius (Standard Deviation of sampled points)
            pos_std = np.mean(np.std(positions, axis=0))

            # Metric 2: Convergence
            mid_idx = n // 2
            best_at_mid = np.min(energies[:mid_idx]) if mid_idx > 0 else energies[0]
            improvement_late = best_at_mid - best_val

            report = [
                "BAYESIAN OPTIMIZATION REPORT",
                "="*40,
                f"Total Samples: {n}",
                f"Best Energy: {best_val:.6f}",
                f"Exploration (Avg StdDev): {pos_std:.4f}",
                "-"*30,
                "DIAGNOSIS:"
            ]

            # Heuristics
            if pos_std < 0.5:
                report += ["⚠️ LOW EXPLORATION: Points clustered tightly."]
                report += ["   -> HINT: Increase 'kappa' (e.g. 5.0) or 'init_points'."]
            else:
                report += ["✅ GOOD EXPLORATION: Algorithm searched wide area."]

            if improvement_late <= 0 and n > 10:
                report += ["⚠️ STALLED: No improvement in second half."]
                report += ["   -> HINT: You might be exploiting too much. Switch to 'simulated_annealing' to fine-tune."]
            else:
                report += [f"✅ LEARNING: Improved by {improvement_late:.4f} in later stages."]

            return "\n".join(report)

        except Exception as e:
            return f"Analysis Error: {e}"