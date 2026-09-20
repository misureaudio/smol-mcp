class ArtifactVaultInspectorToolGv2(Tool):
    """
    A diagnostic tool to map the internal structure of an HDF5 vault.
    It complements ArtifactVaultToolGv9 by providing shape analysis,
    attribute readout, and EMD segmentation parameters.
    """
    name = "artifact_inspector"
    description = (
        "Inspects the internal structure of the HDF5 vault. Returns a tree-like "
        "summary of artifact IDs, dataset shapes, dtypes, and metadata. "
        "Essential for 'scouting' the vault before retrieving large arrays."
    )
    inputs = {
        "vault_filename": {
            "type": "string",
            "description": "The name of the vault file (default: 'research_session.h5').",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, safe_dir: str, vault_filename: str = "research_session.h5"):
        super().__init__()
        self.safe_dir = os.path.abspath(safe_dir)
        self.default_vault = vault_filename
        logging.info(f"ArtifactVaultInspectorToolGv2 initialized.\n*** ✅ ---> Vault storage at '{self.safe_dir}' vault: '{self.default_vault}'")

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

    def forward(self, vault_filename: str = None) -> str:
        target_name = vault_filename if vault_filename else self.default_vault
        vault_path = os.path.join(self.safe_dir, os.path.basename(target_name))

        if not os.path.exists(vault_path):
            return f"Error: Vault file '{target_name}' not found."

        try:
            report = {
                "vault_path": vault_path,
                "artifacts": {}
            }

            with h5py.File(vault_path, 'r') as f:
                def visitor(name, obj):
                    # Path segments
                    parts = name.split('/')
                    root_id = parts[0]

                    if root_id not in report["artifacts"]:
                        report["artifacts"][root_id] = {
                            "artifact_metadata": self._safe_attr_dict(f[root_id].attrs),
                            "datasets": {}
                        }

                    if isinstance(obj, h5py.Dataset):
                        ds_info = {
                            "shape": list(obj.shape),  # Convert tuple to list for JSON
                            "dtype": str(obj.dtype),
                            "attributes": self._safe_attr_dict(obj.attrs)
                        }

                        emd_ctx = self._parse_emd_context(name, obj)
                        if emd_ctx:
                            ds_info["emd_params"] = emd_ctx

                        report["artifacts"][root_id]["datasets"][name] = ds_info

                f.visititems(visitor)

            # Use the custom VaultJSONEncoder to handle int64/float32
            return json.dumps(report, indent=2, cls=VaultJSONEncoder)

        except Exception as e:
            error_msg = f"Inspector error: {type(e).__name__}: {str(e)}"
            logging.error(error_msg)
            return error_msg