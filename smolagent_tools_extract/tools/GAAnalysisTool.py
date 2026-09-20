class GAAnalysisTool(Tool):
    name = "ga_analysis_tool"
    description = """
    Analyzes 'history.csv' from the Genetic Algorithm.
    Diagnoses 'Premature Convergence' by looking at Population Diversity vs Energy.
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
        logging.info("GAAnalysisTool initialized.")

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

            # Parse columns
            gens = np.array([int(r['generation']) for r in rows])
            best_E = np.array([float(r['best_energy']) for r in rows])
            avg_E = np.array([float(r['avg_energy']) for r in rows])
            diversity = np.array([float(r['diversity']) for r in rows])

            n_gens = len(gens)
            final_best = best_E[-1]

            # 1. Identify Convergence Generation (When Diversity drops near zero)
            # Threshold: 1% of initial diversity
            initial_div = diversity[0] if diversity[0] > 0 else 1.0
            conv_threshold = initial_div * 0.01

            # Find first index where diversity < threshold
            conv_indices = np.where(diversity < conv_threshold)[0]
            if len(conv_indices) > 0:
                conv_gen = gens[conv_indices[0]]
                conv_pct = (conv_gen / n_gens) * 100
                is_converged = True
            else:
                conv_gen = n_gens
                conv_pct = 100
                is_converged = False

            # 2. Selection Pressure Analysis
            # If Avg Energy drops much slower than Best Energy, the selection pressure is low.
            # If Avg Energy hugs Best Energy tightly, selection pressure is high (Elitism).
            energy_gap = avg_E - best_E
            avg_gap = np.mean(energy_gap)

            report = [
                "GENETIC ALGORITHM REPORT",
                "="*40,
                f"Generations: {n_gens}",
                f"Final Best Energy: {final_best:.6f}",
                f"Initial Diversity: {initial_div:.4f}",
                f"Final Diversity: {diversity[-1]:.4e}",
                "-"*30,
                "DIAGNOSIS:"
            ]

            # Heuristics
            if is_converged:
                report += [f"✅ CONVERGED: Population collapsed at Gen {conv_gen} ({conv_pct:.1f}%)."]
                if conv_pct < 20:
                    report += ["⚠️ PREMATURE CONVERGENCE: Diversity lost too quickly."]
                    report += ["   -> HINT: Increase 'mutation_rate' or 'pop_size'."]
                else:
                    report += ["   -> Timing looks healthy."]
            else:
                report += ["⚠️ NOT CONVERGED: Population is still diverse."]
                report += ["   -> HINT: Increase 'generations' or 'crossover_rate'."]

            if avg_gap < (initial_div * 0.1):
                report += ["ℹ️ HIGH SELECTIVE PRESSURE: The whole population followed the leader."]
            else:
                report += ["ℹ️ HIGH VARIANCE: Many poor individuals survive (High mutation?)."]

            return "\n".join(report)

        except Exception as e:
            return f"Analysis Error: {e}"