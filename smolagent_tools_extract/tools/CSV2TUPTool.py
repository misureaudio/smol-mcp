class CSV2TUPTool(SafeOpenTool):
    name = "csv_2_tup_tool"
    description = """This is a tool that extracts a tuple from a CSV file.
                     If the file is not in a safe directory it raises an exception."""
    inputs = {
        "file_name": {
            "type": "string",
            "description": "The name of the file to be opened."
        }
    }
    # output_type = "tuple[list[str] | None, list[list[str]] | None]"
    output_type = "object"

    def __init__(self, safe_dir):
        super().__init__(safe_dir)
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("Secure CSV2TUPTool initialized.")

    def forward(self, file_name):
        csv_file = super().forward(file_name, "rt",)
        # Iterate through all the pages and extract text
        tupla = []
        tupla = read_dynamic_csv(csv_file)
        csv_file.close()
        return tupla