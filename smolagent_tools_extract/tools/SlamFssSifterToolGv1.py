class SlamFssSifterToolGv1(Tool):
    name = "slam_fss_sifter"
    description = (
        "High-performance windowed EMD sifter. Loads a signal, applies a Tukey window, "
        "performs EMD, and stashes results in the hdf5_physics_archive automatically. "
        "Designed to prevent context overflow during massive 4M sample sweeps."
    )
    inputs = {
        "audio_dict": {
            "type": "any",
            "description": "The dictionary returned by audio_handler_tool."
        },
        "window_size": {
            "type": "integer",
            "description": "Size of the window (e.g., 32768)."
        },
        "overlap": {
            "type": "integer",
            "description": "Overlap in samples (e.g., 1024)."
        },
        "max_imfs": {
            "type": "integer",
            "description": "Limit for EMD.",
            "nullable": True  # <--- FIX: Required because of default value in forward()
        },
        "vault_prefix": {
            "type": "string",
            "description": "Prefix for artifact IDs.",
            "nullable": True  # <--- FIX: Required because of default value in forward()
        }
    }
    output_type = "string"

    def __init__(self, emd_tool, vault_tool):
        super().__init__()
        self.emd_tool = emd_tool
        self.vault_tool = vault_tool
        logging.info("SlamFssSifterToolGv1 initialized with EMD and Vault tools.")

    def forward(self, audio_dict: any, window_size: int, overlap: int, max_imfs: int = 12, vault_prefix: str = "SLAM-FSS") -> str:
        samples = audio_dict['audio_samples']
        # Step size
        S = window_size - overlap
        # Total number of full windows possible
        num_windows = (len(samples) - overlap) // S

        # Pre-calculate the Tukey window for efficiency
        # alpha is the ratio of tapered section to total window length
        window_func = tukey(window_size, alpha=overlap/window_size)

        logging.info(f"Starting Sifter: {num_windows} windows planned.")

        count = 0
        for i in range(num_windows):
            start = i * S
            end = start + window_size

            # Boundary check
            if end > len(samples):
                break

            # 1. Extract and Window the segment
            segment = samples[start:end] * window_func

            # 2. Prepare the payload for the EMD tool
            seg_dict = {
                "audio_samples": segment,
                "metadata": audio_dict.get('metadata', {})
            }

            # 3. Perform the decomposition via the internal EMD tool instance
            # Note: We call .forward() directly to bypass agent-level overhead
            imfs = self.emd_tool.forward(audio_dict=seg_dict, channel=0, max_imfs=max_imfs)

            # 4. Stash in the Vault with the high-quality metadata and hints
            stash_id = f"{vault_prefix}_{i:03d}"
            self.vault_tool.forward(
                action='stash',
                artifact_id=stash_id,
                data={'original_signal': segment, 'imf_matrix': imfs},
                access_hint=(
                    f"Hierarchical Radix-2 data for window {i}. "
                    "result['data']['original_signal'] is the windowed time signal. "
                    "result['data']['imf_matrix'] holds the modes on axis 1 (Index 0 = IMF 1). "
                    "DO NOT re-run EMD on these IMFs."
                ),
                metadata={'window_index': i, 'start_sample': start}
            )
            count += 1
            if count % 10 == 0:
                logging.info(f"Sifter Progress: {count}/{num_windows} windows vaulted.")

        return f"MISSION SUCCESS: {count} windows processed and stashed with prefix '{vault_prefix}'."