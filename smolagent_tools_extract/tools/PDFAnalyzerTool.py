class PDFAnalyzerTool(Tool):
    name = "pdf_analyzer_tool"
    description = """Analyzes a PDF file to determine the best text extraction method (direct vs. OCR).
                     It provides a detailed report on text content, images, and text readability.
                     The file must be located in the secure upload directory."""
    inputs = {
        "pdf_filename": {
            "type": "string",
            "description": "The name of the PDF file (e.g., 'report.pdf') located in the safe upload directory."
        },
        "min_readable_ratio": {
            "type": "number",  # <-- FIX IS HERE
            "description": "The minimum ratio of readable characters (0.0 to 1.0) to consider text directly extractable. Defaults to 0.7.",
            "nullable": True
        },
        "sample_pages": {
            "type": "integer",
            "description": "The number of pages to sample for a quick analysis of large documents. Defaults to 3.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, safe_dir: str):
        super().__init__()
        self.safe_dir = safe_dir
        if not os.path.isdir(self.safe_dir):
            os.makedirs(self.safe_dir, exist_ok=True)
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("Secure PDFAnalyzerTool initialized.")

    def _sample_pages(self, total_pages, sample_pages_count):
        if total_pages <= sample_pages_count:
            return list(range(total_pages))
        step = total_pages / sample_pages_count
        return [int(i * step) for i in range(sample_pages_count)]

    def _calculate_readable_ratio(self, text):
        if not text:
            return 0.0
        readable_pattern = r'[a-zA-Z0-9\s\.,;:!?\-\'\"()\[\]{}/@#$%&*+=<>|~`]'
        readable_chars = len(re.findall(readable_pattern, text))
        total_chars = len(text)
        return readable_chars / total_chars if total_chars > 0 else 0.0

    def _recommend_method(self, results):
        has_text = results["has_text"]
        has_images = results["has_images"]
        is_readable = results["text_is_readable"]
        avg_readable = results["avg_readable_ratio"]

        if not has_text and has_images:
            return "OCR_REQUIRED", 0.95
        if not has_text and not has_images:
            return "EMPTY_OR_UNSUPPORTED", 0.95
        if has_text and is_readable:
            confidence = 0.95 if not has_images else 0.85
            return "DIRECT_EXTRACTION", confidence
        if has_text and not is_readable:
            if avg_readable < 0.3:
                return "OCR_REQUIRED", 0.90
            else:
                return "HYBRID", 0.70
        return "HYBRID", 0.50

    def _format_report(self, results):
        report_lines = []
        if "error" in results:
            report_lines.append("=" * 70)
            report_lines.append("PDF ANALYSIS ERROR")
            report_lines.append("=" * 70)
            report_lines.append(f"\nError: {results['error']}")
            if "path" in results:
                report_lines.append(f"Path: {results['path']}")
            report_lines.append("=" * 70)
            return "\n".join(report_lines)

        report_lines.append("=" * 70)
        report_lines.append("PDF TEXT EXTRACTION ANALYSIS REPORT")
        report_lines.append("=" * 70)
        report_lines.append("\nDocument Info:")
        report_lines.append(f"  Total Pages: {results['total_pages']}")
        report_lines.append(f"  Pages Sampled: {results['sampled_pages']}")
        report_lines.append("\nContent Detection:")
        report_lines.append(f"  Has Text Layer: {results['has_text']}")
        report_lines.append(f"  Has Images: {results['has_images']}")
        report_lines.append(f"  Text is Readable: {results['text_is_readable']}")
        report_lines.append(f"  Avg Readable Ratio: {results['avg_readable_ratio']:.2%}")

        report_lines.append("\nPer-Page Analysis:")
        for page in results['page_analysis']:
            report_lines.append(f"  Page {page['page_num']}:")
            report_lines.append(f"    Text Length: {page['text_length']} chars")
            report_lines.append(f"    Has Text: {page['has_text']}")
            report_lines.append(f"    Has Images: {page['has_images']}")
            report_lines.append(f"    Readable Ratio: {page['readable_ratio']:.2%}")

        report_lines.append("\n" + "=" * 70)
        report_lines.append(f"RECOMMENDATION: {results['recommended_method']}")
        report_lines.append(f"Confidence: {results['confidence']:.0%}")
        report_lines.append("=" * 70)

        method = results['recommended_method']
        report_lines.append("\nNext Steps:")
        if method == 'DIRECT_EXTRACTION':
            report_lines.append("  ✓ Use direct text extraction (e.g., pdf_2_txt_tool).")
        elif method == 'OCR_REQUIRED':
            report_lines.append("  ✓ Use OCR tools. Direct text extraction will likely fail.")
        elif method == 'HYBRID':
            report_lines.append("  ⚠ This PDF is mixed. Try direct extraction first, but be prepared to use OCR if results are poor.")
        else:
            report_lines.append("  ⚠ Document may be empty, corrupted, or use an unsupported format.")

        return "\n".join(report_lines)

    def forward(self, pdf_filename: str, min_readable_ratio: float = 0.7, sample_pages: int = 3):
        full_path = os.path.abspath(os.path.join(self.safe_dir, os.path.basename(pdf_filename)))
        safe_dir_abs = os.path.abspath(self.safe_dir)

        if not full_path.startswith(safe_dir_abs):
            return self._format_report({"error": f"Access denied. File '{pdf_filename}' is outside the safe directory.", "path": full_path})

        if not os.path.exists(full_path):
            return self._format_report({"error": "File not found", "path": full_path})

        try:
            doc = fitz.open(full_path)
        except Exception as e:
            return self._format_report({"error": f"Error opening PDF: {str(e)}", "path": full_path})

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return self._format_report({"error": "PDF document is empty (0 pages).", "path": full_path})

        page_indices = self._sample_pages(total_pages, sample_pages)

        results = {
            "total_pages": total_pages,
            "sampled_pages": len(page_indices),
            "has_text": False,
            "has_images": False,
            "text_is_readable": False,
            "avg_readable_ratio": 0.0,
            "page_analysis": [],
        }

        total_readable_ratio = 0.0
        for page_num in page_indices:
            page = doc.load_page(page_num)
            page_text = page.get_text()
            has_images = len(page.get_images(full=True)) > 0
            has_text = len(page_text.strip()) > 0

            results["has_text"] |= has_text
            results["has_images"] |= has_images

            readable_ratio = self._calculate_readable_ratio(page_text) if has_text else 0.0
            total_readable_ratio += readable_ratio

            results["page_analysis"].append({
                "page_num": page_num,
                "text_length": len(page_text),
                "has_text": has_text,
                "has_images": has_images,
                "readable_ratio": readable_ratio
            })

        doc.close()

        if len(page_indices) > 0:
            results["avg_readable_ratio"] = total_readable_ratio / len(page_indices)

        results["text_is_readable"] = results["avg_readable_ratio"] >= min_readable_ratio

        results["recommended_method"], results["confidence"] = self._recommend_method(results)

        return self._format_report(results)