class SyncAlignmentToolV2(Tool):
    name = "sync_alignment_tool"
    description = (
        "Precision alignment tool. Uses the SECOND Dirac pulse as a Master Anchor "
        "to avoid boundary effects at the signal origin. Corrects manual alignment "
        "errors between Ref (L) and DUT (R)."
    )
    inputs = {
        "artifact_id": {"type": "string", "description": "Stereo artifact ID (L=Ref, R=DUT)."},
        "search_window_ms": {"type": "number", "description": "Search radius (e.g., 10.0ms).", "nullable": True}
    }
    output_type = "any"

    def __init__(self, vault_tool):
        super().__init__()
        self.vault = vault_tool
        logging.info(f"SyncAlignmentToolV2 initialized: {vault_tool}")

    def forward(self, artifact_id: str, search_window_ms: float = 10.0) -> dict:
        # 1. Retrieve Data
        data = self.vault.forward(action="retrieve", artifact_id=artifact_id)
        samples = data['samples']
        Fs = data['metadata'].get('sample_rate', 192000)
        ref_sig = samples[:, 0]
        dut_sig = samples[:, 1]

        # 2. Locate the first TWO anchors in Reference (L)
        # We use a simple peak detector with a distance constraint (min 0.5s between pulses)
        peaks, _ = find_peaks(np.abs(ref_sig), height=np.max(np.abs(ref_sig))*0.8, distance=int(0.5*Fs))

        if len(peaks) < 2:
            return {"status": "ERROR", "message": "Could not find two Dirac anchors in Reference signal."}

        # MASTER ANCHOR: The second pulse pair (index 1)
        ref_master_idx = peaks[1]

        # 3. Define the Stretching-Safe Search Zone in DUT (R)
        # 10ms search radius around the Reference anchor
        radius = int((search_window_ms / 1000.0) * Fs)
        search_start = ref_master_idx - radius
        search_end = ref_master_idx + radius

        # SARD-style safety check
        if search_start < 0:
            search_start = 0

        dut_zone = dut_sig[search_start:search_end]

        # 4. Cross-Correlation (Fine Alignment)
        # We use a 20-sample kernel from the reference pulse
        ref_kernel = ref_sig[ref_master_idx-10:ref_master_idx+10]
        corr = np.correlate(dut_zone, ref_kernel, mode='valid')

        # Locate the exact peak in the capture
        fine_lag_in_zone = np.argmax(corr)
        actual_dut_idx = search_start + fine_lag_in_zone + 10

        # 5. Result: The 'Golden Offset'
        static_lag = actual_dut_idx - ref_master_idx

        results = {
            "ref_anchor_sample": int(ref_master_idx),
            "dut_anchor_sample": int(actual_dut_idx),
            "static_lag_samples": int(static_lag),
            "static_lag_ms": float((static_lag / Fs) * 1000.0),
            "alignment_confidence": float(np.max(corr) / (np.linalg.norm(ref_kernel) * np.linalg.norm(dut_zone[fine_lag_in_zone:fine_lag_in_zone+20])))
        }

        # 6. Stash for future Agent logic
        self.vault.forward(
            action="stash",
            artifact_id=f"ALIGN_{artifact_id}",
            data=results,
            access_hint="Static alignment offset. Always add 'static_lag_samples' when slicing the DUT channel."
        )

        return results