class PDFSplitterToolGv1(SafeOpenTool):
    name = "pdf_splitter_tool"
    description = """This tool extracts text from a PDF (local or URL) and splits it into Markdown files.
                     It preserves formatting and tables. Files are saved in the safe directory."""
    inputs = {
        "file_name": {
            "type": "string",
            "description": "The pathname or URL of the PDF file to be split."
        },
        "pages_per_chunk": {
            "type": "integer",
            "description": "Maximum number of pages per generated Markdown file."
        }
    }
    output_type = "string"

    def __init__(self, safe_dir):
        super().__init__(safe_dir)
        logging.info("Secure PDFSplitterToolGv1 initialized.")

    def forward(self, file_name, pages_per_chunk):
        # 1. Handle Input (Logic mirrored from PDF2TXTTool)
        if os.path.isfile(file_name) or not re.match(valid_url_pattern, file_name):
            # SafeOpenTool handles path stripping and security logic
            binary_file = super().forward(file_name, "rb")
        elif re.match(valid_url_pattern, file_name) is not None:
            # Handle Web URL
            logging.info(f"PDFSplitterTool: Downloading PDF from {file_name}")
            with httpx.stream("GET", file_name) as response:
                binary_file = io.BytesIO(b"".join(response.iter_raw()))
        else:
            raise ValueError(f"The string {file_name} is neither a web URL nor a valid path.")

        try:
            # 2. Open the PDF from the binary stream
            # We read() the stream because fitz/pymupdf4llm needs a seekable buffer or bytes
            pdf_bytes = binary_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)

            # Use a clean base name for output files
            clean_base = os.path.splitext(os.path.basename(file_name))[0]
            created_files = []

            # 3. Split into Markdown chunks
            chunk_count = 1
            for start_page in range(0, total_pages, pages_per_chunk):
                end_page = min(start_page + pages_per_chunk, total_pages)
                page_range = list(range(start_page, end_page))

                # pymupdf4llm can accept the fitz.Document object directly
                md_text = pymupdf4llm.to_markdown(doc, pages=page_range)

                # Construct output path within safe_dir
                out_name = f"{clean_base}_part{chunk_count}_p{start_page+1}-{end_page}.md"
                out_path = os.path.join(self.safe_dir, out_name)

                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(md_text)

                created_files.append(out_name)
                chunk_count += 1

            doc.close()
            binary_file.close()

            return f"Successfully split '{clean_base}' into {len(created_files)} Markdown files: {', '.join(created_files)}"

        except Exception as e:
            if 'binary_file' in locals():
                binary_file.close()
            return f"Error splitting PDF: {str(e)}"