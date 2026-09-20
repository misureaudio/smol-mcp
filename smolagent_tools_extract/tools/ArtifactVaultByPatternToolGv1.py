class ArtifactVaultByPatternToolGv1(Tool):
    name = "artifact_by_pattern"
    description = (
        "Inspects the HDF5 vault. If 'artifact_id' is provided, it returns deep shapes "
        "and metadata for that specific item. If 'artifact_id' is omitted, it returns "
        "a high-level summary of all available IDs to save context space."
    )
    inputs = {
        "artifact_id": {
            "type": "string",
            "description": "Optional: Specific ID or prefix (e.g., 'MLSESS_IMFs_Seg_1*') to inspect.",
            "nullable": True
        },
        "vault_filename": {
            "type": "string",
            "description": "The name of the vault file.",
            "nullable": True
        }
    }
    output_type = "string"

    # ... (Keep __init__, _safe_attr_dict, _parse_emd_context, and VaultJSONEncoder) ...
    def __init__(self, safe_dir: str, vault_filename: str = "research_session.h5"):
        super().__init__()
        self.safe_dir = os.path.abspath(safe_dir)
        self.default_vault = vault_filename
        logging.info(f"ArtifactVaultByPatternToolGv1 initialized.\n*** ✅ ---> Vault storage at '{self.safe_dir}' vault: '{self.default_vault}'")

    def _safe_attr_dict(self, attrs):
        """Converts h5py attributes to a JSON-serializable dict."""
        out = {}
        for k, v in attrs.items():
            # Convert bytes to string if necessary
            if isinstance(v, bytes):
                out[k] = v.decode('utf-8')
            else:
                out[k] = v
        return out

    def _parse_emd_context(self, name, obj):
        """Identifies EMD-specific patterns (k, m, IMF index)."""
        context = {}
        # Check attributes
        for attr_key in ['k', 'm', 'segment_id', 'imf_number']:
            if attr_key in obj.attrs:
                context[attr_key] = obj.attrs[attr_key]

        # Regex fallback for paths like 'seg_k12_m8/IMF_3'
        k_m_match = re.search(r'k(\d+)_m(\d+)', name)
        if k_m_match:
            context['k'] = int(k_m_match.group(1))
            context['m'] = int(k_m_match.group(2))

        imf_match = re.search(r'IMF[_-]?(\d+)', name, re.IGNORECASE)
        if imf_match:
            context['imf_number'] = int(imf_match.group(1))

        return context

    def forward(self, artifact_id: str = None, vault_filename: str = None) -> str:
        target_name = vault_filename if vault_filename else self.default_vault
        vault_path = os.path.join(self.safe_dir, os.path.basename(target_name))

        if not os.path.exists(vault_path):
            return f"Error: Vault file '{target_name}' not found."

        try:
            with h5py.File(vault_path, 'r') as f:
                # CASE 1: High-level Catalog (No specific ID requested)
                if not artifact_id:
                    all_ids = list(f.keys())
                    return json.dumps({
                        "vault": target_name,
                        "total_artifacts": len(all_ids),
                        "available_ids": all_ids,
                        "usage_tip": "To see shapes/metadata, call artifact_inspector(artifact_id='NAME')"
                    }, indent=2)

                # CASE 2: Selective Inspection (Regex/Wildcard support)
                report = {"vault": target_name, "artifacts": {}}

                # Convert glob-style '*' to regex
                search_pattern = artifact_id.replace('*', '.*')
                regex = re.compile(f"^{search_pattern}$")

                def visitor(name, obj):
                    root_id = name.split('/')[0]
                    if regex.match(root_id):
                        if root_id not in report["artifacts"]:
                            # Fetch metadata once for the root artifact
                            root_metadata = self._safe_attr_dict(f[root_id].attrs)
                            report["artifacts"][root_id] = {
                                "metadata": root_metadata,
                                "datasets": {}
                            }

                        if isinstance(obj, h5py.Dataset):
                            # 1. Basic Dataset Info
                            ds_info = {
                                "shape": list(obj.shape),
                                "dtype": str(obj.dtype),
                                "emd_params": self._parse_emd_context(name, obj)
                            }

                            # 2. ALIGNMENT LOGIC (Protocol Detection)
                            # We check the root metadata AND the ID string for robust detection
                            root_meta = report["artifacts"][root_id]["metadata"]
                            is_audio_type = (
                                root_meta.get('type') == 'audio_metadata' or
                                'sample_rate' in root_meta or
                                "audio" in root_id.lower() or
                                "Fs" in root_id
                            )

                            if is_audio_type:
                                ds_info["protocol_compatibility"] = "AUDIO_V1"
                                ds_info["usage_hint"] = "Direct pipe to sifter/emd_tool enabled (dict includes 'audio_samples')."

                            # 3. Assign the consolidated info to the report
                            report["artifacts"][root_id]["datasets"][name] = ds_info

                f.visititems(visitor)

                if not report["artifacts"]:
                    return f"No artifacts found matching pattern: {artifact_id}"

                return json.dumps(report, indent=2, cls=VaultJSONEncoder)

        except Exception as e:
            return f"Inspector error: {str(e)}"