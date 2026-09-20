class PDFSplitterToolGv2(SafeOpenTool):
    name = "pdf_splitter_tool"
    description = """Splits a PDF into Markdown files. It ensures chunks do not break
                     in the middle of a paragraph by splitting at double newlines."""
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
        logging.info("Secure PDFSplitterToolGv2 initialized.")

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
            doc = fitz.open(stream=binary_file.read(), filetype="pdf")
            total_pages = len(doc)
            clean_base = os.path.splitext(os.path.basename(file_name))[0]

            all_created_files = []
            chunk_count = 1

            # This buffer will hold the "hanging" paragraph text from the previous chunk
            carry_over_text = ""

            for start_page in range(0, total_pages, pages_per_chunk):
                end_page = min(start_page + pages_per_chunk, total_pages)
                page_range = list(range(start_page, end_page))

                # Extract current range
                current_md = pymupdf4llm.to_markdown(doc, pages=page_range)

                # Prepend any text that was cut off from the previous iteration
                full_text = carry_over_text + current_md

                # If this is not the last chunk, we find the last logical break
                if end_page < total_pages:
                    # Look for the last double newline (paragraph break)
                    last_break = full_text.rfind("\n\n")

                    if last_break != -1:
                        # Split: 'content' goes to file, 'carry_over' goes to next loop
                        content = full_text[:last_break].strip()
                        carry_over_text = full_text[last_break:].strip() + "\n\n"
                    else:
                        # If no double newline found, keep all and clear carry_over
                        content = full_text
                        carry_over_text = ""
                else:
                    # Last chunk: take everything remaining
                    content = full_text

                # Save the content
                out_name = f"{clean_base}_part{chunk_count}_p{start_page+1}-{end_page}.md"
                out_path = os.path.join(self.safe_dir, out_name)

                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(content)

                all_created_files.append(out_name)
                chunk_count += 1

            doc.close()
            return f"Created {len(all_created_files)} semantic chunks: {', '.join(all_created_files)}"

        except Exception as e:
            return f"Error: {str(e)}"