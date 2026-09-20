class SAAnalysisTool(Tool):
    name = "sa_analysis_tool"
    description = """
    Advanced analysis of Simulated Annealing history CSV.
    Detects convergence issues, acceptance problems, stagnation patterns,
    and gives precise, actionable tuning recommendations (v3 — fully upgraded).
    Supports all v2+ features: adaptive scaling, restarts, temperature logging.
    """
    inputs = {
        "csv_filename": {
            "type": "string",
            "description": "Name of the history CSV file (e.g., 'run1.csv')"
        }
    }
    output_type = "string"

    def __init__(self, safe_dir):
        super().__init__()
        self.safe_dir = safe_dir
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("SAAnalysisTool initialized.")

    def forward(self, csv_filename: str):
        if not csv_filename.lower().endswith('.csv'):
            return "Error: File must end with .csv"

        path = os.path.join(self.safe_dir, os.path.basename(csv_filename))
        if not os.path.exists(path):
            return f"Error: File not found: {path}"

        try:
            with open(path, 'r') as f:
                reader = csv.DictReader(f, delimiter=';')
                rows = list(reader)

            if len(rows) < 50:
                return "Not enough data for meaningful analysis (<50 iterations)."

            # Parse required columns
            iters = np.array([int(r['iteration']) for r in rows])
            energy = np.array([float(r['energy']) for r in rows])
            best_energy = np.array([float(r['best_so_far']) for r in rows])  # recommended to log
            accepted = np.array([int(r.get('accepted', 1)) for r in rows])  # fallback 1
            temperature = np.array([float(r.get('temperature', 1.0)) for r in rows])
            scale = np.array([float(r.get('neighbor_scale', 0.1)) for r in rows])

            n = len(iters)
            final_best = best_energy[-1]
            best_idx = np.argmin(best_energy)
            conv_pct = best_idx / n * 100

            report = [
                "SIMULATED ANNEALING ANALYSIS REPORT (v3)",
                "="*60,
                f"Total iterations       : {n:,}",
                f"Final best energy      : {final_best:.10f}",
                f"Best found at iter     : {best_idx} ({conv_pct:.1f}% of run)",
                f"Initial temperature    : {temperature[0]:.2f}",
                f"Final temperature      : {temperature[-1]:.2e}",
                f"Final neighbor scale   : {scale[-1]:.2e}",
                "",
            ]

            # 1. Acceptance Rate Analysis
            accept_rate = accepted.mean() * 100
            late_accept = accepted[-max(100, n//10):].mean() * 100
            report += [
                "ACCEPTANCE RATE",
                "-"*30,
                f"Overall acceptance     : {accept_rate:.1f}%",
                f"Late-phase acceptance  : {late_accept:.1f}%",
            ]
            if accept_rate < 15:
                report += ["Too low! Steps too small or temperature too low early."]
                report += ["→ Enable or keep adapt_neighbor=True"]
                report += ["→ Try higher initial_temperature"]
            elif accept_rate > 70:
                report += ["Too high! Too random, wasting evaluations."]
                report += ["→ Lower initial_temperature"]
                report += ["→ Use faster cooling (e.g. cooling_rate=0.90)"]
            else:
                report += ["Healthy range (15–70%). Good exploration/exploitation balance."]

            # 2. Convergence & Quenching
            report += ["", "CONVERGENCE BEHAVIOR", "-"*30]
            if conv_pct < 8:
                report += ["QUENCHING: Best found too early → cooled too fast!"]
                report += ["→ Use slower schedule: 'log', 'cauchy', or cooling_rate ≥ 0.99"]
                report += ["→ Or enable 'adaptive' schedule"]
            elif conv_pct > 90:
                report += ["SLOW CONVERGENCE: Still improving at the end."]
                report += ["→ Increase max_iterations by 50–100%"]
            else:
                report += ["Convergence timing looks good."]

            # 3. Stagnation & Restart Analysis
            report += ["", "STAGNATION & RESTART NEEDS", "-"*30]
            stagnant_periods = []
            current_stag = 0
            for i in range(1, n):
                if best_energy[i] >= best_energy[i-1]:
                    current_stag += 1
                else:
                    if current_stag > 200:
                        stagnant_periods.append((i - current_stag, current_stag))
                    current_stag = 0
            if current_stag > 200:
                stagnant_periods.append((n - current_stag, current_stag))

            if stagnant_periods:
                longest = max(stagnant_periods, key=lambda x: x[1])
                report += [f"Detected {len(stagnant_periods)} stagnation period(s)"]
                report += [f"Longest: {longest[1]} iterations (from {longest[0]})"]
                suggested = max(300, longest[1] // 2)
                report += [f"STRONG RECOMMENDATION: Enable restart_stagnation = {suggested}"]
                report += ["Best modes: 'reheat' (safe), 'random_jump' (aggressive)"]
            else:
                report += ["No major stagnation — excellent run!"]

            # 4. Adaptive Scaling Effectiveness
            if np.std(scale) > 1e-8:  # scale actually changed
                report += ["", "ADAPTIVE SCALING", "-"*30]
                if 0.3 < late_accept < 0.6:
                    report += ["Adaptive scaling worked perfectly!"]
                    report += [f"Final scale {scale[-1]:.2e} is a good starting point for future runs."]
                else:
                    report += ["Adaptive scaling tried but didn't converge to ideal rate."]
                    report += ["→ Try larger adapt_window (e.g. 200) or keep running longer."]
            else:
                report += ["Adaptive scaling was disabled."]

            # 5. Final Recommendations Summary
            report += ["", "ACTION PLAN", "="*60]
            recs = []
            if 'QUENCHING' in "".join(report):
                recs += ["Use schedule_type='log' or cooling_rate=0.995+"]
            if stagnant_periods:
                recs += [f"Set restart_stagnation={max(300, longest[1]//2)}"]
            if accept_rate > 70:
                recs += ["Lower initial_temperature by 50%"]
            if n < 20000 and conv_pct > 80:
                recs += ["Increase max_iterations to 30k–50k"]
            if not recs:
                recs = ["No major issues! This was a strong run. Consider minor scaling up."]

            report += recs
            report += ["", "Tip: Rerun with these changes and compare best energy!"]

            return "\n".join(report)

        except Exception as e:
            logging.exception("SAAnalysisTool failed")
            return f"Analysis failed: {str(e)}"