class ArtifactVaultInspectorToolGv3(Tool):
    """
    A diagnostic 'scout' for the HDF5 vault.
    Personality 1 (Default): Returns a lite list of artifact IDs.
    Personality 2 (Targeted): Returns deep shapes/metadata for specific IDs or patterns.

    IMPORTANT: The output of a 'targeted' search can be large.
    Assign the result to a variable; DO NOT print the full output in your final answer.
    """
    name = "artifact_inspector"
    description = (
        "Inspects the vault. If 'artifact_id' is omitted, returns a lite list of IDs. "
        "If 'artifact_id' is provided, returns deep metadata, shapes, and protocol hints "
        "for matching items. Use this to scout data before retrieval."
    )
    inputs = {
        "artifact_id": {
            "type": "string",
            "description": "Optional: Specific ID or wildcard (e.g., 'EMD*') for deep inspection.",
            "nullable": True
        },
        "vault_filename": {
            "type": "string",
            "description": "Vault filename (defaults to 'research_session.h5').",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, safe_dir: str, vault_filename: str = "research_session.h5"):
        super().__init__()
        self.safe_dir = os.path.abspath(safe_dir)
        self.default_vault = vault_filename
        logging.info(f"ArtifactVaultInspectorToolGv3 initialized.\n*** ✅ ---> Vault storage at '{self.safe_dir}' vault: '{self.default_vault}'")

    def forward(self, artifact_id: str = None, vault_filename: str = None) -> str:
        target_name = vault_filename if vault_filename else self.default_vault
        vault_path = os.path.join(self.safe_dir, os.path.basename(target_name))

        if not os.path.exists(vault_path):
            return f"Error: Vault '{target_name}' not found."

        try:
            with h5py.File(vault_path, 'r') as f:
                # --- PERSONALITY 1: THE LITE CATALOG ---
                if not artifact_id:
                    ids = list(f.keys())
                    return json.dumps({
                        "vault": target_name,
                        "status": "LITE_CATALOG",
                        "count": len(ids),
                        "available_ids": ids,
                        "instruction": "For deep details, call: artifact_inspector(artifact_id='ID_NAME')"
                    }, indent=2)

                # --- PERSONALITY 2: THE TARGETED EXPERT ---
                report = {"vault": target_name, "status": "DEEP_INSPECTION", "artifacts": {}}

                # Regex for wildcard support
                regex = re.compile(f"^{artifact_id.replace('*', '.*')}$")

                def visitor(name, obj):
                    root_id = name.split('/')[0]
                    if regex.match(root_id):
                        if root_id not in report["artifacts"]:
                            # Fetch root metadata
                            meta = {k: (v.decode('utf-8') if isinstance(v, bytes) else v)
                                    for k, v in f[root_id].attrs.items()}

                            report["artifacts"][root_id] = {
                                "metadata": meta,
                                "datasets": {}
                            }

                        if isinstance(obj, h5py.Dataset):
                            # Determine Protocol Alignment
                            is_audio = (
                                report["artifacts"][root_id]["metadata"].get('type') == 'audio_metadata' or
                                'sample_rate' in report["artifacts"][root_id]["metadata"] or
                                "audio" in root_id.lower() or "Fs" in root_id
                            )

                            ds_info = {
                                "shape": list(obj.shape),
                                "dtype": str(obj.dtype),
                                "emd_context": self._parse_emd_context(name, obj)
                            }

                            if is_audio:
                                ds_info["protocol"] = "AUDIO_V1"
                                ds_info["hint"] = "Compatible with emd_tool/sifter via 'audio_samples' key."

                            report["artifacts"][root_id]["datasets"][name] = ds_info

                f.visititems(visitor)

                if not report["artifacts"]:
                    return f"No artifacts matching '{artifact_id}'"

                return json.dumps(report, indent=2, cls=VaultJSONEncoder)

        except Exception as e:
            return f"Inspector error: {str(e)}"

    def _parse_emd_context(self, name, obj):
        """Standard EMD parameter extractor."""
        ctx = {k: obj.attrs[k] for k in ['k', 'm', 'segment_id', 'imf_number'] if k in obj.attrs}
        if not ctx:  # Regex fallback
            km = re.search(r'k(\d+)_m(\d+)', name)
            if km:
                ctx['k'], ctx['m'] = int(km.group(1)), int(km.group(2))
        return ctx