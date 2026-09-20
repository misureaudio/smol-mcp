class MarkdownToPDFTool(Tool):
    name = "markdown_to_pdf_tool"
    description = """Converts a given Markdown string into a PDF file.
                     It allows specifying the paper size, page orientation, and margin.
                     The resulting PDF is saved in a secure directory."""
    inputs = {
        "markdown_content": {
            "type": "string",
            "description": "The Markdown content to be converted into a PDF."
        },
        "output_filename": {
            "type": "string",
            "description": "The desired filename for the output PDF (e.g., 'report.pdf'). It must end with .pdf."
        },
        "paper_format": {
            "type": "string",
            "description": "The paper size. Common values are 'a4', 'a3', 'letter', 'legal', 'a5'. Defaults to 'a4'.",
            "nullable": True
        },
        "orientation": {
            "type": "string",
            "description": "The page orientation. Must be either 'portrait' or 'landscape'. Defaults to 'portrait'.",
            "nullable": True
        },
        "margin": {
            "type": "string",
            "description": "The page margin as a string (e.g., '1in', '2cm', '0.75in'). Defaults to '1in'.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, safe_dir: str):
        super().__init__()
        self.safe_dir = safe_dir
        # Ensure the underlying md2pdf module is imported
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        try:
            # from md2pdf import md2pdf
            # self.md2pdf_func = md2pdf
            # from md2pdfcp import md2pdf_cp
            # self.md2pdf_func = md2pdf_cp
            from md2pdflx import md2pdf_lx
            self.md2pdf_func = md2pdf_lx
            logging.info("Secure MarkdownToPDFTool initialized.")
        except ImportError:
            # raise ImportError("The 'md2pdf.py' module could not be found. Ensure md2pdf-Claude-Pandoc.py is renamed to md2pdf.py and is in the same directory.")
            # raise ImportError("The 'md2pdfcp.py' module could not be found. Ensure md2pdfcp.py is in the same directory.")
            logging.info("Secure MarkdownToPDFTool initialization failed.")
            raise ImportError("The 'md2pdflx.py' module could not be found. Ensure md2pdflx.py is in the same directory.")

    def forward(self, markdown_content: str, output_filename: str, paper_format: str = 'a4', orientation: str = 'portrait', margin: str = '1in'):
        if not output_filename.lower().endswith('.pdf'):
            return f"Error: The output_filename '{output_filename}' must end with .pdf."

        full_path = os.path.join(self.safe_dir, os.path.basename(output_filename))

        try:
            print(f"Generating PDF with settings: paper='{paper_format}', orientation='{orientation}', margin='{margin}'", file=sys.stderr)

            # Call the md2pdf function with all parameters
            output_pdf_bytes = self.md2pdf_func(
                input_md=markdown_content,
                orientation=orientation,
                paper_format=paper_format,
                margin=margin
            )

            # Write the resulting bytes to the specified file
            with open(full_path, 'wb') as pdf_file:
                pdf_file.write(output_pdf_bytes)

            return f"Successfully created PDF: '{full_path}' with the specified formatting."

        except Exception as e:
            error_message = f"An error occurred while creating the PDF: {e}"
            logging.error(error_message)
            return f"Error: {error_message}"