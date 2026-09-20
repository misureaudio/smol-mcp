class ArtifactVaultToolGv10(Tool):
    name = "artifact_vault"
    description = (
        "A persistent HDF5 repository. Supports 'stash', 'retrieve', 'catalog', and 'update'. "
        "Retrieval is 'Flattened': for complex artifacts (like EMD), you can access signals "
        "directly via keys like result['original_signal'] or result['imf_matrix']."
    )
    # ... (inputs and __init__ remain same as Gv9) ...
    inputs = {
        "action": {
            "type": "string",
            "description": "Operation: 'stash', 'retrieve', 'catalog', or 'update'."
        },
        "artifact_id": {
            "type": "string",
            "description": "Unique ID for the artifact.",
            "nullable": True
        },
        "data": {
            "type": "any",
            "description": "Data to stash (Numpy array). Not needed for 'update'.",
            "nullable": True
        },
        "access_hint": {
            "type": "string",
            "description": "Instructional Manual: What the data represents and how to index it.",
            "nullable": True
        },
        "metadata": {
            "type": "object",
            "description": "Dictionary of parameters/annotations.",
            "nullable": True
        },  # BEGIN Mod MATTIA 20260113
        "vault_filename": {
            "type": "string",
            "description": "Vault filename (defaults to 'research_session.h5').",
            "nullable": True
        }  # END Mod MATTIA 20260113
    }
    output_type = "any"

    def __init__(self,
                 safe_dir: str,
                 vault_filename: str = "research_session.h5"):
        super().__init__()
        self.safe_dir = os.path.abspath(safe_dir)
        if not os.path.exists(self.safe_dir):
            os.makedirs(self.safe_dir,
                        exist_ok=True)
        self.default_vault = os.path.basename(vault_filename)
        self.vault_path = os.path.join(self.safe_dir,
                                       self.default_vault)
        logging.info(f"ArtifactVaultToolGv10 initialized.\n*** ✅ ---> Vault storage at '{self.safe_dir}' vault: '{self.default_vault}'")

    def forward(self, action: str, artifact_id: str = None, data: any = None,
                access_hint: str = None, metadata: dict = None, vault_filename: str = None) -> any:
        '''
        if vault_filename:
            vault_path = os.path.join(self.safe_dir, os.path.basename(vault_filename))
        else:
            vault_path = self.vault_path
        if not os.path.exists(vault_path):
            return f"Error: Vault '{target_name}' not found."
        '''

        target_name = vault_filename if vault_filename else self.default_vault
        vault_path = os.path.join(self.safe_dir, os.path.basename(target_name))

        if not os.path.exists(vault_path):
            return f"Error: Vault file '{target_name}' not found."

        action = action.lower()
        try:
            if action == 'stash':
                if not artifact_id or data is None:
                    return "Error: ID and data required."

                with h5py.File(self.vault_path, 'a') as f:
                    if artifact_id in f:
                        del f[artifact_id]
                    group = f.create_group(artifact_id)

                    # --- UPGRADE: HIERARCHICAL DICT SUPPORT ---
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, np.ndarray):
                                group.create_dataset(key, data=value, compression="gzip", compression_opts=4)
                            else:
                                group.create_dataset(key, data=str(value))
                    elif isinstance(data, np.ndarray):
                        group.create_dataset('raw_data', data=data, compression="gzip", compression_opts=4)
                    else:
                        group.create_dataset('raw_data', data=str(data))

                    group.attrs['timestamp'] = str(datetime.now())
                    group.attrs['user_instruction'] = str(access_hint) if access_hint else "N/A"
                    if metadata:
                        for k, v in metadata.items():
                            group.attrs[k] = v

                    return {"status": "SUCCESS", "passport": self._generate_passport(artifact_id, data, access_hint)}

            # ... [Stash logic remains Gv8 hierarchical] ...
            # --- ACTION: RETRIEVE (Aligned & Flattened Gv10) ---
            if action == 'retrieve':
                if not artifact_id:
                    return "Error: ID required."
                with h5py.File(self.vault_path, 'r') as f:
                    if artifact_id not in f:
                        return f"Error: '{artifact_id}' not found."
                    group = f[artifact_id]

                    # 1. Extract Internal Data
                    keys = list(group.keys())
                    stored_meta = dict(group.attrs)
                    user_manual = stored_meta.pop('user_instruction', "N/A")

                    # 2. Build the Base Response Envelope
                    response = {
                        "metadata": stored_meta,
                        "instructional_manual": user_manual,
                        "vault_id": artifact_id
                    }

                    # 3. SMART FLATTENING LOGIC
                    # CASE A: Simple Artifact (Single 'raw_data' array)
                    if len(keys) == 1 and keys[0] == 'raw_data':
                        response["data"] = group['raw_data'][:]
                        # Aligned Protocol for Audio
                        if 'sample_rate' in stored_meta:
                            response["audio_samples"] = response["data"]
                            response["technical_bridge"] = "AUDIO_V1: Use result['audio_samples']."
                        else:
                            response["technical_bridge"] = "SIMPLE: Use result['data']."

                    # CASE B: Complex/Hierarchical Artifact (e.g., EMD result)
                    else:
                        # Promote all internal keys (imf_matrix, original_signal, etc.) to top level
                        for k in keys:
                            response[k] = group[k][:]

                        # Aligned Protocol for Audio/EMD
                        if 'original_signal' in response:
                            response["audio_samples"] = response['original_signal']
                            response["technical_bridge"] = (
                                f"FLATTENED_AUDIO: Keys {keys} are available at top level. "
                                "Use result['original_signal'] or result['audio_samples']."
                            )
                        else:
                            response["technical_bridge"] = f"FLATTENED_COMPLEX: Access keys {keys} directly from result."

                    return response

            # ... (Rest of stash/catalog/update logic) ...
            # --- ACTION: CATALOG ---
            elif action == 'catalog':
                if not os.path.exists(self.vault_path):
                    return {"available_artifacts": []}
                with h5py.File(self.vault_path, 'r') as f:
                    return {"available_artifacts": list(f.keys())}

        except Exception as e:
            return f"Vault error: {str(e)}"

    def _generate_passport(self, id, data, hint):
        p = {"id": id, "instructional_manual": hint if hint else "N/A", "timestamp": str(datetime.now())}
        if hasattr(data, 'shape'):
            p["shape"] = list(data.shape)
        return p