class AudioHandlerTool(Tool):
    name = "audio_handler_tool"
    description = """This tool loads and saves .wav or .flac audio files within designated secure directories.

    Operations:
    - 'read': Returns a dictionary containing 'audio_samples' (as a numpy array) and 'metadata'.
      Reads strictly from the READ directory.
    - 'write': Saves a numpy array of audio samples to a file with specific quantization.
      Writes strictly to the WRITE directory.

    IMPORTANT: The 'read' output contains raw massive arrays. NEVER try to print the return value
    of this tool as an agent action or in the final answer. Always assign it to a variable."""

    inputs = {
        "file_name": {
            "type": "string",
            "description": "The name of the file (wav/flac). Directory components are stripped."
        },
        "action": {
            "type": "string",
            "description": "Either 'read' or 'write'."
        },
        "audio_data": {
            "type": "any",
            "description": "For 'write' mode only. A numpy array containing the audio samples.",
            "nullable": True
        },
        "sample_rate": {
            "type": "integer",
            "description": "For 'write' mode only. The sampling rate of the audio.",
            "nullable": True
        },
        "quantization": {
            "type": "integer",
            "description": "Bit depth for saving files. Allowed: [16, 24, 32] for .wav, [16, 24] for .flac.",
            "nullable": True
        }
    }
    output_type = "any"

    def __init__(self, safe_dir_read="upload", safe_dir_write="uplaud"):
        super().__init__()
        self.safe_dir_read = safe_dir_read
        self.safe_dir_write = safe_dir_write

        # --- FIX 1: Corrected Logging Variables ---

        # Ensure the read directory exists
        if not os.path.exists(self.safe_dir_read):
            try:
                os.makedirs(self.safe_dir_read)
                logging.info(f"Created safe read directory: '{os.path.abspath(self.safe_dir_read)}'")
            except OSError as e:
                logging.error(f"Failed to create safe read directory: {e}")

        # Ensure the write directory exists
        if not os.path.exists(self.safe_dir_write):
            try:
                os.makedirs(self.safe_dir_write)
                # FIX: In your PDF, this line printed self.safe_dir_read instead of write
                logging.info(f"Created safe write directory: '{os.path.abspath(self.safe_dir_write)}'")
            except OSError as e:
                logging.error(f"Failed to create safe write directory: {e}")

        logging.info(f"AudioHandlerTool initialized.\n*** ✅ ---> READ source: '{os.path.abspath(self.safe_dir_read)}'\n*** ✅ ---> WRITE dest: '{os.path.abspath(self.safe_dir_write)}'")

    def forward(self, file_name, action, audio_data=None, sample_rate=None, quantization=None):
        # 1. Sanitize
        clean_name = os.path.basename(file_name)

        # Validate extension upfront
        _, ext = os.path.splitext(clean_name)
        ext = ext.lower()
        if ext not in ['.wav', '.flac']:
            raise ValueError(f"Unsupported file format '{ext}'. Only .wav and .flac are allowed.")

        # --- FIX 2: Compute paths only for the specific action to avoid confusion ---

        if action == 'read':
            # Construct Read Path
            file_path = os.path.join(self.safe_dir_read, clean_name)
            abs_file_path = os.path.abspath(file_path)
            abs_safe_dir = os.path.abspath(self.safe_dir_read)

            # Security Check (Read)
            if not abs_file_path.startswith(abs_safe_dir):
                raise ValueError(f"Access denied: Path {abs_file_path} is outside read dir {abs_safe_dir}")

            try:
                if not os.path.exists(abs_file_path):
                    raise FileNotFoundError(f"File not found: {clean_name} (looked in {abs_safe_dir})")

                data, samplerate = sf.read(abs_file_path)
                channels = 1 if len(data.shape) == 1 else data.shape[1]

                logging.info(f"AudioHandlerTool: Read '{clean_name}' successfully.")
                return {
                    "metadata": {
                        "filename": clean_name,
                        "sample_rate": samplerate,
                        "channels": channels,
                        "format": ext.replace('.', '').upper(),
                        "frames": len(data)
                    },
                    "audio_samples": data
                }
            except Exception as e:
                raise RuntimeError(f"Error reading audio file: {str(e)}")

        elif action == 'write':
            # Construct Write Path
            file_path = os.path.join(self.safe_dir_write, clean_name)
            abs_file_path = os.path.abspath(file_path)
            abs_safe_dir = os.path.abspath(self.safe_dir_write)

            # Security Check (Write)
            if not abs_file_path.startswith(abs_safe_dir):
                raise ValueError(f"Access denied: Path {abs_file_path} is outside write dir {abs_safe_dir}")

            if audio_data is None or sample_rate is None:
                raise ValueError("Arguments 'audio_data' and 'sample_rate' are required for write action.")

            # Subtype logic (same as your PDF)
            subtype = None
            if quantization is not None:
                if ext == '.flac':
                    if quantization not in [16, 24]:
                        raise ValueError(f"Invalid quantization {quantization} for .flac. Allowed: [16, 24].")
                    subtype = f'PCM_{quantization}'
                elif ext == '.wav':
                    if quantization not in [16, 24, 32]:
                        raise ValueError(f"Invalid quantization {quantization} for .wav. Allowed: [16, 24, 32].")
                    subtype = f'PCM_{quantization}'

            try:
                sf.write(abs_file_path, audio_data, sample_rate, subtype=subtype)
                msg = f"Successfully saved audio to {clean_name}"
                if subtype:
                    msg += f" with {subtype} encoding"
                logging.info(f"AudioHandlerTool: {msg}")
                return msg
            except Exception as e:
                raise RuntimeError(f"Error writing audio file: {str(e)}")

        else:
            raise ValueError(f"Unknown action '{action}'. Please use 'read' or 'write'.")