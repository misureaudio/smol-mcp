class SafeOpenTool(Tool):
    name = "safe_open_tool"
    description = """This is a tool that opens a file replacing open().
                     It enforces security by forcing file access to occur strictly within the designated safe directory.
                     Any directory paths provided in 'file_name' will be stripped."""
    inputs = {
        "file_name": {
            "type": "string",
            "description": "The name of the file to be opened. directory components are ignored; the file is looked up in the safe directory."
        },
        "mode": {
            "type": "string",
            "description": "'rt' for text files, 'rb' for binary files (pdf, docx...)"
        }
    }
    output_type = "object"

    def __init__(self, safe_dir):
        super().__init__()
        self.safe_dir = safe_dir
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        # logging.info("Secure SafeOpenTool initialized.")
        logging.info(f"SafeOpenTool initialized.\n*** ✅ ---> Files will be saved in '{os.path.abspath(self.safe_dir)}'")

    def forward(self, file_name, mode):
        # 1. Sanitize: Ignore whatever path the agent provided and force usage of basename
        #    Example: 'uploads/run1.csv' -> 'run1.csv'
        #    Example: 'C:/Windows/System32/run1.csv' -> 'run1.csv'
        clean_name = os.path.basename(file_name)

        # 2. Construct the forced path inside the safe directory
        file_path = os.path.join(self.safe_dir, clean_name)
        abs_file_path = os.path.abspath(file_path)
        abs_safe_dir = os.path.abspath(self.safe_dir)

        # 3. Security Verification (Double check)
        #    Ensure the resulting path is actually inside the safe dir
        if not abs_file_path.startswith(abs_safe_dir):
            raise ValueError(f"Access denied: File path {abs_file_path} is outside the allowed path {abs_safe_dir}")
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        # logging.info(f"SavePlotTool initialized.\n*** ✅ ---> Plots will be saved in '{os.path.abspath(self.safe_dir)}'")
        logging.info(f"SafeOpenTool: Opening '{abs_file_path}' in mode '{mode}'")

        try:
            if 'b' in mode:
                return open(abs_file_path, mode)
            else:
                # Default to text mode with utf-8 encoding
                return open(abs_file_path, mode, encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {clean_name} (looked in {abs_safe_dir})")