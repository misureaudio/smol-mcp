class TFDDataTool(Tool):
    name = "tfd_data_tool"
    description = """
    Manages the storage and retrieval of raw Time-Frequency Analysis data.
    - 'save': Saves the TFD data dictionary to a compressed .npz file.
    - 'load': Loads a .npz file back into a dictionary for processing.
    """
    inputs = {
        "action": {
            "type": "string",
            "description": "'save' or 'load'"
        },
        "filename": {
            "type": "string",
            "description": "The filename (e.g., 'analysis_results'). Extension .npz added automatically."
        },
        "data_dict": {
            "type": "any",
            "description": "The dictionary returned by TFDTool (required for 'save').",
            "nullable": True
        }
    }
    output_type = "any"

    def __init__(self, safe_dir="tfd_data"):
        super().__init__()
        self.safe_dir = safe_dir
        if not os.path.exists(self.safe_dir):
            os.makedirs(self.safe_dir, exist_ok=True)
        logging.info("TFDDataTool v1 (Gv1) initialized. Time/Frequency Distributions data will be saved.")

    def forward(self, action, filename, data_dict=None):
        try:
            clean_name = os.path.basename(filename).replace('.npz', '')
            file_path = os.path.join(self.safe_dir, f"{clean_name}.npz")

            if action == 'save':
                if not data_dict:
                    return "Error: 'data_dict' is required for save action."

                # We use savez_compressed. We must unpack the dict to kwargs.
                # Note: We cast complex matrices to save space if needed,
                # but numpy handles complex128 native well.
                np.savez_compressed(file_path, **data_dict)
                return f"TFD Data successfully saved to '{file_path}'"

            elif action == 'load':
                if not os.path.exists(file_path):
                    return f"Error: File '{file_path}' not found."

                # Load back into a dictionary
                loaded = np.load(file_path, allow_pickle=True)
                # Convert NpzFile object back to a standard dict
                result_dict = {key: loaded[key] for key in loaded.files}

                # If metadata was saved as a 0-d array object (common in save_z), extract it
                if 'metadata' in result_dict and result_dict['metadata'].ndim == 0:
                    result_dict['metadata'] = result_dict['metadata'].item()
                elif 'axes' in result_dict and result_dict['axes'].ndim == 0:
                    result_dict['axes'] = result_dict['axes'].item()

                return result_dict

            else:
                return "Error: Action must be 'save' or 'load'."

        except Exception as e:
            return f"TFDDataTool Error: {str(e)}"