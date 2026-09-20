class ListBPFunctionsTool(Tool):
    name = "list_bulletproof_functions"
    description = """
    Lists all available standard optimization functions stored in the internal BULLETPROOF library.
    Use this to discover available test functions (like 'mogadish', 'schwefel', 'cross_in_tray')
    before running an optimizer.
    """
    inputs = {
        "verbose": {
            "type": "boolean",
            "description": "If True (default), returns details (bounds, global min, description). If False, returns only names.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("ListBPFunctionTool initialized.")

    def forward(self, verbose: bool = True):
        # Access the global dictionary safely
        # This looks for the variable defined in your main script
        bp_dict = globals().get('BULLETPROOF_FUNCTIONS', {})

        if not bp_dict:
            return "Error: The BULLETPROOF_FUNCTIONS library is empty or could not be found in the global scope."

        if verbose is None:
            verbose = True

        report = ["📚 **AVAILABLE OPTIMIZATION FUNCTIONS** 📚\n"]

        for name, data in bp_dict.items():
            if not verbose:
                report.append(f"- {name}")
            else:
                # Extract Metadata
                bounds = data.get('bounds', 'Custom')
                gmin = data.get('global_min', 'Unknown')

                # Attempt to extract a docstring/summary from the code source
                # We look for the first comment line inside the code string
                code_snippet = data.get('code', '')
                description = "No description provided."
                for line in code_snippet.split('\n'):
                    stripped = line.strip()
                    if stripped.startswith('#') and 'def' not in stripped and 'import' not in stripped:
                        description = stripped.lstrip('# ').strip()
                    break  # Take the first comment as the description

                report.append(f"🔹 **{name}**")
                report.append(f"   Description: {description}")
                report.append(f"   Bounds: {bounds}")
                report.append(f"   Known Global Min: {gmin}")
                report.append("")

        return "\n".join(report)