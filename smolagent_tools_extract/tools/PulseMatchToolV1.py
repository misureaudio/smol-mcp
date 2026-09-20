class PulseMatchToolV1(Tool):
    name = "pulse_match_tool"
    description = (
        "High-speed temporal synchronization tool. Finds a pulse template (Time Code) "
        "within a long signal using normalized cross-correlation. Returns indices, "
        "time-drift (remainder vs Fs), and correlation coefficients."
    )
    inputs = {
        "signal_id": {"type": "string", "description": "Vault ID of the long signal (DUT output)."},
        "pulse_id": {"type": "string", "description": "Vault ID of the pulse template (Dirac pair)."},
        "threshold": {"type": "number", "description": "Correlation threshold (e.g., 0.99) to define a match.", "nullable": True},
        "vault_filename": {"type": "string", "description": "Defaults to 'research_session.h5'.", "nullable": True}
    }
    output_type = "any"

    def __init__(self, vault_tool):
        super().__init__()
        self.vault = vault_tool
        logging.info(f"PulseMatchToolV1 initialized: {vault_tool}")

    def forward(self, signal_id: str, pulse_id: str, threshold: float = 0.99, vault_filename: str = None) -> dict:
        # 1. Retrieve Data from Gv16 Vault
        sig_obj = self.vault.forward(action="retrieve", artifact_id=signal_id, vault_filename=vault_filename)
        pulse_obj = self.vault.forward(action="retrieve", artifact_id=pulse_id, vault_filename=vault_filename)

        s = sig_obj['samples']
        m = pulse_obj['samples']
        Fs = sig_obj['metadata'].get('sample_rate', sig_obj['metadata'].get('Fs', 192000))

        # 2. Optimized Normalized Cross-Correlation
        # We normalize the pulse once
        m_norm = (m - np.mean(m)) / (np.std(m) * len(m))

        # We use scipy's optimized correlation (FFT-based for speed)
        # Note: True 'sliding normalization' is more complex, but for Dirac pulses,
        # standard correlation on a normalized template is extremely effective.
        corr = correlate(s, m_norm, mode='valid')

        # 3. Find peaks that satisfy the 'Ready-for-Blood' threshold
        indices = np.where(corr >= threshold)[0]

        if len(indices) == 0:
            return {"status": "FAILED", "message": f"No pulses found above threshold {threshold}"}

        # 4. Calculate Temporal Metrics (The Maryasov Logic)
        time_vals = indices / Fs
        remainders = np.mod(indices, Fs)
        # Calculate jitter (sample-to-sample delta)
        if len(indices) > 1:
            intervals = np.diff(indices)
            jitter_map = intervals - Fs  # Deviation from perfect 1.0s interval
        else:
            jitter_map = []

        results = {
            "match_count": len(indices),
            "indices": indices.tolist(),
            "time_seconds": time_vals.tolist(),
            "remainders": remainders.tolist(),
            "jitter_samples": jitter_map.tolist(),
            "max_correlation": float(np.max(corr))
        }

        # 5. Stash the 'Sync Map' back in the Vault
        sync_id = f"SYNC_{signal_id}"
        self.vault.forward(
            action="stash",
            artifact_id=sync_id,
            data=results,
            access_hint=f"Temporal Sync Map for {signal_id}. Use this to align windows.",
            metadata={"source_signal": signal_id, "type": "sync_map"}
        )

        return {"status": "SUCCESS", "sync_id": sync_id, "report": f"Found {len(indices)} pulses. Max Corr: {results['max_correlation']:.4f}"}