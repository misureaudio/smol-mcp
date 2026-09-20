class PDF2TXTTool(SafeOpenTool):
    name = "pdf_2_txt_tool"
    description = """This is a tool that extracts the text from a PDF file.
                     If the file is not in a safe directory it raises an exception."""
    inputs = {
        "file_name": {
            "type": "string",
            "description": "The pathname or URL of the PDF file to be opened."
        }
    }
    output_type = "string"

    def __init__(self, safe_dir):
        super().__init__(safe_dir)
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("Secure PDF2TXTTool initialized.")

    def forward(self, file_name):
        if os.path.isfile(file_name):
            # if it's a path to a local file:
            # binary_file = super().forward(file_name, "rb") # invoke SafeOpenTool
            binary_file = super().forward(file_name, "rb")  # invoke SafeOpenTool
        elif re.match(valid_url_pattern, file_name) is not None:
            # else, if it's a web link:
            with httpx.stream("GET", file_name) as response:
                binary_file = io.BytesIO(b"".join(response.iter_raw()))
        else:
            raise ValueError(f"The string {file_name} is neither a web URL nor a path to a local file")

        # Iterate through all the pages and extract text
        text = ""
        import fitz
        with fitz.open("pdf", binary_file.read()) as pdf:
            # Iterate through all the pages and extract text
            for page_number in range(len(pdf)):
                page = pdf[page_number]
                page_text = page.get_text()  # Extract text from the current page
                # print(f"****** New page of {file_name}:\n{page_text}")
                text += page_text
        binary_file.close()
        return text