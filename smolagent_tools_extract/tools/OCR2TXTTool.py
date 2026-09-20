class OCR2TXTTool(Tool):
    name = "ocr_2_txt_tool"
    description = """Extracts text from PDF files using Optical Character Recognition (OCR).
                     This tool is essential for scanned documents, PDFs containing only images, or when PDFAnalyzerTool recommends 'OCR_REQUIRED' or 'HYBRID'.
                     It is more resource-intensive than the standard pdf_2_txt_tool.
                     The file must be located in the secure upload directory."""
    inputs = {
        "file_name": {
            "type": "string",
            "description": "The name of the PDF file (e.g., 'scanned_report.pdf') located in the safe upload directory."
        },
        "language": {
            "type": "string",
            "description": "The language of the text in the document (e.g., 'eng' for English, 'ita' for Italian, 'fra' for French). Defaults to 'eng'.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, safe_dir: str):
        super().__init__()
        self.safe_dir = os.path.abspath(safe_dir)
        if not os.path.isdir(self.safe_dir):
            os.makedirs(self.safe_dir, exist_ok=True)
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("Secure OCR2TXTTool initialized.")

        # --- Proactive Dependency Check during Initialization ---
        tesseract_path = shutil.which('tesseract')
        if not tesseract_path:
            raise ImportError(
                "Tesseract is not installed or not in your PATH. "
                "OCR2TXTTool cannot function without it. "
                "Please install Tesseract-OCR for your system."
            )
        logging.info(f"Tesseract executable found at: {tesseract_path}")

        # pdf2image depends on Poppler
        pdftoppm_path = shutil.which('pdftoppm')
        if not pdftoppm_path:
            raise ImportError(
                "Poppler is not installed or not in your PATH. "
                "OCR2TXTTool depends on pdf2image, which requires Poppler. "
                "Please install poppler-utils (on Linux) or Poppler for your system."
            )
        logging.info(f"Poppler (pdftoppm) executable found at: {pdftoppm_path}")

    def forward(self, file_name: str, language: str = 'eng'):
        # --- Security: Sanitize and validate the file path ---
        full_path = os.path.abspath(os.path.join(self.safe_dir, os.path.basename(file_name)))

        if not full_path.startswith(self.safe_dir):
            return f"Error: Access denied. File '{file_name}' is outside the safe directory."

        if not os.path.exists(full_path):
            return f"Error: File not found at '{full_path}'."

        try:
            logging.info(f"Starting OCR process for '{full_path}' with language '{language}'.")
            # Convert PDF to a list of PIL images
            images = convert_from_path(full_path)

            full_text = []
            for i, image in enumerate(images):
                logging.info(f"  - Performing OCR on page {i + 1}/{len(images)}")
                # Use pytesseract to extract text from the image
                text = pytesseract.image_to_string(image, lang=language)
                full_text.append(text)

            logging.info(f"Successfully completed OCR for '{full_path}'.")
            # Join the text from all pages
            return "\n\n--- Page Break ---\n\n".join(full_text)

        except Exception as e:
            error_message = f"An error occurred during the OCR process: {str(e)}"
            logging.error(error_message)
            return f"Error: {error_message}"