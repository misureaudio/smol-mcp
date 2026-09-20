class EMDTool(Tool):
    """
    A specialized tool to perform Empirical Mode Decomposition (EMD) on a signal.
    It takes the standard audio dictionary (from AudioHandlerTool) and returns the Intrinsic Mode Functions (IMFs).
    """
    name = "emd_tool"
    description = (
        "Performs Empirical Mode Decomposition (EMD) on a specific audio channel. "
        "Returns a 2D NumPy array of the Intrinsic Mode Functions (IMFs). "
        "The output shape is (number_of_samples, number_of_imfs). "
    )
    inputs = {
        "audio_dict": {
            "type": "any",
            "description": "The dictionary returned by AudioHandlerTool (containing 'audio_samples' and 'metadata')."
        },
        "channel": {
            "type": "integer",
            "description": "The channel index to analyze if audio is multi-channel. Default is 0.",
            "nullable": True
        },
        "max_imfs": {
            "type": "integer",
            "description": "Optional: The maximum number of IMFs to extract.",
            "nullable": True
        }
    }
    output_type = "any"  # Returns a NumPy array

    def __init__(self):
        super().__init__()
        # No internal logging config here to avoid overwriting global Agent logging
        logging.info("EMDTool initialized. Empirical Mode Decomposition will be computed.")

    def forward(self, audio_dict, channel=0, max_imfs=None) -> np.ndarray:
        """
        Executes the EMD process using the 'emd' library with unified input handling.
        """
        try:
            # --- 1. Validation and Extraction (Unified Logic) ---
            if not isinstance(audio_dict, dict) or 'audio_samples' not in audio_dict or 'metadata' not in audio_dict:
                return "Error: Input must be the dictionary output from AudioHandlerTool."

            samples = audio_dict['audio_samples']
            metadata = audio_dict['metadata']
            logging.info(f"EMDTOOl debug: {metadata}")
            # fs is extracted but not strictly needed for basic Sift, useful if we extend to Masked Sift later
            fs = metadata.get('sample_rate', None)

            # --- 2. Channel Selection (Nd -> 1D) ---
            if samples.ndim > 1:
                num_channels = samples.shape[1]
                if channel >= num_channels:
                    return f"Error: Channel {channel} requested, but audio only has {num_channels} channels."
                sig = samples[:, channel]
                logging.info(f"EMDTool: Extracted channel {channel}/{num_channels}")
            else:
                sig = samples
                logging.info("EMDTool: Processing Mono signal.")

            # --- 3. EMD Execution ---
            # Call the real, underlying library function
            imfs = emd.sift.sift(sig, max_imfs=max_imfs)

            # Sanity check on output
            if imfs.shape[0] != len(sig):
                raise RuntimeError(f"EMD process returned an unexpected shape: {imfs.shape}. Expected {len(sig)} samples.")

            logging.info(f"EMDTool: Decomposition complete. Extracted {imfs.shape[1]} IMFs.")
            return {
                'imf_matrix': imfs,
                'sample_rate': fs
                }

        except Exception as e:
            error_msg = f"Error during EMD processing: {type(e).__name__}: {e}"
            logging.error(error_msg)
            return error_msg