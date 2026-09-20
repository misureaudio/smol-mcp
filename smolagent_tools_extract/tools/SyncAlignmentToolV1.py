class SyncAlignmentToolV1(Tool):
    name = "sync_alignment_tool"
    description = (
        "Corrects manual alignment errors between MATLAB reference (L) and ADC capture (R). "
        "Finds the precise sample lag and returns a 'Realigned' view of the signal."
    )
    inputs = {
        "artifact_id": {"type": "string", "description": "The stereo artifact ID (L=Ref, R=DUT)."},
        "search_window_ms": {"type": "number", "description": "Window to search for alignment (e.g., 5.0 for 5ms).", "nullable": True}
    }
    output_type = "any"

    def __init__(self, vault_tool):
        super().__init__()
        self.vault = vault_tool
        logging.info(f"SyncAlignmentToolV1 initialized: {vault_tool}")

    def forward(self, artifact_id: str, search_window_ms: float = 5.0) -> dict:
        # 1. Retrieve the stereo data
        data = self.vault.forward(action="retrieve", artifact_id=artifact_id)
        samples = data['samples']  # Shape (N, 2)
        Fs = data['metadata'].get('sample_rate', 192000)

        ref_sig = samples[:, 0]
        dut_sig = samples[:, 1]

        # 2. Find the first Coarse Anchor in the Reference (L)
        # We look for the Dirac pulse (max amplitude)
        ref_anchor_idx = np.argmax(np.abs(ref_sig[:Fs*2]))  # Search first 2 seconds

        # 3. Define the Search Window in the DUT (R)
        win_size = int((search_window_ms / 1000.0) * Fs)
        search_start = max(0, ref_anchor_idx - win_size)
        search_end = min(len(dut_sig), ref_anchor_idx + win_size)

        dut_search_zone = dut_sig[search_start:search_end]

        # 4. Perform Fine Cross-Correlation
        # We find where the DUT pulse best matches the Ref pulse
        # Using a small slice of the ref signal around its peak
        ref_slice = ref_sig[ref_anchor_idx-10:ref_anchor_idx+10]
        corr = np.correlate(dut_search_zone, ref_slice, mode='valid')

        # The 'fine_lag' is relative to the start of our search zone
        fine_lag_zone_idx = np.argmax(corr)
        actual_dut_anchor_idx = search_start + fine_lag_zone_idx + 10  # adjust for slice offset

        # 5. Calculate the Systemic Offset
        static_lag = actual_dut_anchor_idx - ref_anchor_idx

        results = {
            "artifact_id": artifact_id,
            "sample_rate": Fs,
            "static_lag_samples": int(static_lag),
            "static_lag_ms": float((static_lag / Fs) * 1000.0),
            "alignment_quality": float(np.max(corr))
        }

        # 6. Stash an 'Alignment Passport'
        self.vault.forward(
            action="stash",
            artifact_id=f"ALIGN_{artifact_id}",
            data=results,
            access_hint="Alignment data. Use static_lag_samples to slice DUT data correctly."
        )

        return results