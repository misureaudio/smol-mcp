class SlamFssSifterToolGv3(Tool):
    name = "slam_fss_sifter"
    description = (
        "High-performance windowed EMD sifter. (S)elf-(A)ware of (R)ussian (D)olls: "
        "automatically unpacks nested audio dicts. If the signal is too short for multiple "
        "windows, it processes a single full-length window."
    )
    inputs = {
        "audio_dict": {
            "type": "any",
            "description": "The dictionary returned by audio_handler_tool or artifact_vault."
        },
        "window_size": {
            "type": "integer",
            "description": "Size of the window (e.g., 131072)."
        },
        "overlap": {
            "type": "integer",
            "description": "Overlap in samples (e.g., 4096)."
        },
        "max_imfs": {
            "type": "integer",
            "description": "Limit for EMD.",
            "nullable": True
        },
        "vault_prefix": {
            "type": "string",
            "description": "Prefix for artifact IDs.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, emd_tool, vault_tool):
        super().__init__()
        self.emd_tool = emd_tool
        self.vault_tool = vault_tool
        logging.info("SARD Sifter Gv3 initialized. Unpacking protection active.")

    def forward(self, audio_dict: any, window_size: int, overlap: int, max_imfs: int = 12, vault_prefix: str = "SLAM-FSS") -> str:
        # 1. THE SARD UNPACKING LOGIC
        # Handle the case where the agent wraps a vault result inside another 'audio_samples' key
        payload = audio_dict.get('audio_samples')

        # Check if the payload itself is another dictionary (The Russian Doll)
        if isinstance(payload, dict):
            logging.info("Sifter: Russian Doll detected! Unpacking nested audio_dict...")
            metadata = payload.get('metadata', {})
            samples = payload.get('audio_samples')
        else:
            # Standard Gv11 flattened structure
            samples = payload
            metadata = audio_dict.get('metadata', {})

        if samples is None or not hasattr(samples, "__len__"):
            return "Error: No valid audio array found. Check your input structure."

        N = len(samples)

        # 2. THE "RIEN À FAIRE" CURE (Single Window Logic)
        # If the signal is shorter than the requested window, we adjust to process what we have
        if N < window_size:
            logging.info(f"Sifter: Signal ({N}) shorter than window ({window_size}). Using N as window size.")
            window_size = N
            overlap = 0

        S = window_size - overlap
        if S <= 0:
            return "Error: Overlap must be smaller than window_size."

        # Calculate number of windows, ensuring at least 1 if we have samples
        num_windows = max(1, (N - overlap) // S)

        # Final boundary safety: don't plan a window that exceeds N
        if (num_windows - 1) * S + window_size > N:
            num_windows = max(0, num_windows - 1)

        logging.info(f"Starting Sifter: {num_windows} windows planned for {N} samples.")
        if num_windows == 0:
            return "Error: Signal is too short for any processing."

        # 3. THE SIFTING LOOP
        window_func = tukey(window_size, alpha=overlap/window_size) if window_size > 0 else 1.0
        count = 0

        for i in range(num_windows):
            start = i * S
            end = start + window_size

            if end > N:
                break  # Safety break

            # 1. Prepare segment
            segment = samples[start:end] * window_func

            # 2. Extract incoming sample_rate to pass it down
            incoming_fs = metadata.get('sample_rate', metadata.get('Fs'))

            seg_dict = {
                "audio_samples": segment,
                "metadata": {"sample_rate": incoming_fs}  # PROPAGATION
            }

            # 3. Call EMD and handle DICTIONARY return
            emd_result = self.emd_tool.forward(audio_dict=seg_dict, channel=0, max_imfs=max_imfs)

            if isinstance(emd_result, str):  # Handle error messages
                return emd_result

            imfs = emd_result['imf_matrix']
            # Prioritize EMD's returned FS, fallback to incoming
            final_fs = emd_result.get('sample_rate', incoming_fs)

            # 4. Stash in Vault with EXPLICIT metadata for Gv12 Protocol
            stash_id = f"{vault_prefix}_{i:03d}"
            self.vault_tool.forward(
                action='stash',
                artifact_id=stash_id,
                data={'samples': segment, 'imf_matrix': imfs},
                metadata={
                    'window_index': i,
                    'start_sample': start,
                    'type': 'audio_metadata',  # Activates AUDIO_V1 logic
                    'sample_rate': final_fs  # Essential for metrology
                }
            )

            count += 1
            if count % 10 == 0:
                logging.info(f"Sifter Progress: {count}/{num_windows}")

        return f"MISSION SUCCESS: {count} windows vaulted with prefix '{vault_prefix}'."