class GetBPFunctionCodeTool(Tool):
    name = "get_bulletproof_function_code"
    description = """
    Retrieves the raw Python source code for a specific function from the BULLETPROOF library.
    Use this when you need to inspect the code, modify it (e.g., convert 'numpy' logic to 'mpmath'),
    or pass it to a tool that doesn't support automatic lookup.
    """
    inputs = {
        "function_name": {
            "type": "string",
            "description": "The name of the function to retrieve (e.g., 'mogadish', 'rastrigin')."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("GetBPFunctionCodeTool initialized.")

    def forward(self, function_name: str):
        bp_dict = globals().get('BULLETPROOF_FUNCTIONS', {})

        clean_name = function_name.strip().lower()

        if clean_name not in bp_dict:
            return f"Error: Function '{function_name}' not found in the library. Use 'list_bulletproof_functions' to see available options."

        data = bp_dict[clean_name]
        code = data.get('code', '')

        if not code:
            return f"Error: No source code found for '{function_name}'."

        return code