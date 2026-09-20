class ArtifactVaultToolGv16(Tool):
    name = "artifact_vault"
    description = (
        "A unified HDF5 repository for DSP data. Supports four actions:\n"
        "- 'catalog': Returns a lite list of all artifact IDs.\n"
        "- 'inspect': Returns deep metadata and shapes for a specific ID or wildcard pattern (e.g. 'EMD*').\n"
        "- 'retrieve': Returns the data. Complex artifacts (EMD) are 'Flattened' for direct access to signals.\n"
        "- 'stash': Saves numpy arrays or dictionaries with metadata into the vault."
        "- 'audit': Scans all artifacts matching a prefix and returns a summary of their dataset shapes. "
        "   Use 'audit' for global complexity analysis without looping manually."
    )
    inputs = {
        "action": {
            "type": "string",
            "description": "Operation: 'catalog', 'inspect', 'retrieve', or 'stash'."
        },
        "artifact_id": {
            "type": "string",
            "description": "Unique ID or pattern (required for retrieve, stash, inspect).",
            "nullable": True
        },
        "data": {
            "type": "any",
            "description": "Data to stash (Numpy array or dict).",
            "nullable": True
        },
        "access_hint": {
            "type": "string",
            "description": "Stash only: Instructions on what the data represents.",
            "nullable": True
        },
        "metadata": {
            "type": "object",
            "description": "Stash only: Dictionary of annotations.",
            "nullable": True
        },
        "vault_filename": {
            "type": "string",
            "description": "Vault filename (defaults to 'research_session.h5').",
            "nullable": True
        }
    }
    output_type = "any"

    def __init__(self, safe_dir: str,
                 inspector_tool: Tool,
                 vault_filename: str = "research_session.h5"):
        super().__init__()
        self.safe_dir = os.path.abspath(safe_dir)
        if not os.path.exists(self.safe_dir):
            os.makedirs(self.safe_dir, exist_ok=True)
        self.default_vault = os.path.basename(vault_filename)
        self.vault_path = os.path.join(self.safe_dir, self.default_vault)

        # NESTING: We store the inspector tool internally
        self.inspector = inspector_tool
        logging.info(f"ArtifactVaultToolGv16 initialized.\n*** ✅ ---> Vault storage at '{self.safe_dir}' vault: '{self.default_vault}'")

    def forward(self, action: str,
                artifact_id: str = None,
                data: any = None,
                access_hint: str = None,
                metadata: dict = None,
                vault_filename: str = None) -> any:

        # 1. Resolve which vault file we are hitting
        target_name = vault_filename if vault_filename else self.default_vault
        active_vault_path = os.path.join(self.safe_dir, os.path.basename(target_name))

        if not os.path.exists(active_vault_path) and action.lower() != 'stash':
            return f"Error: Vault file '{target_name}' not found."

        action = action.lower()
        try:
            # --- ACTION: CATALOG & INSPECT (Delegated to Scout) ---
            if action in ['catalog', 'inspect']:
                # The inspector handles its own file opening
                # BEGIN Gv12
                return json.loads(
                    self.inspector.forward(
                        artifact_id=artifact_id if action == 'inspect' else None,
                        vault_filename=target_name
                        )
                    )
            # END Gv12
            # BEGIN Gv14
            # --- NEW ACTION: AUDIT (Internal Scavenging) ---
            elif action == 'audit':
                if not artifact_id:
                    return "Error: pattern/prefix required for 'audit'."

                # Convert glob '*' to regex '.*'
                search_pattern = artifact_id.replace('*', '.*')
                regex = re.compile(f"^{search_pattern}$")

                matches_found = {}  # Use explicit names

                with h5py.File(active_vault_path, 'r') as f:
                    for entry_id in f.keys():
                        if regex.match(entry_id):
                            # Scavenge specifically for IMF matrices
                            group = f[entry_id]
                            if 'imf_matrix' in group:
                                ds = group['imf_matrix']
                                if hasattr(ds, 'shape'):
                                    matches_found[entry_id] = list(ds.shape)

                    if not matches_found:
                        return f"Audit found no artifacts matching: {artifact_id}"

                    # --- ALGORITHMIC ANALYSIS (No more 's' ambiguity) ---
                    all_imf_counts = []
                    for shape_tuple in matches_found.values():
                        if len(shape_tuple) > 1:
                            all_imf_counts.append(shape_tuple[1])

                    if not all_imf_counts:
                        return "Audit found artifacts, but none contained valid IMF matrices."

                    abs_min = min(all_imf_counts)
                    abs_max = max(all_imf_counts)

                    min_id_list = []
                    max_id_list = []
                    for eid, shp in matches_found.items():
                        if shp[1] == abs_min:
                            min_id_list.append(eid)
                        if shp[1] == abs_max:
                            max_id_list.append(eid)

                    return {
                        "vault": target_name,
                        "match_count": len(matches_found),
                        "complexity_report": {
                            "min_imfs_found": abs_min,
                            "max_imfs_found": abs_max,
                            "simplest_artifact_ids": min_id_list,
                            "most_complex_artifact_ids": max_id_list
                        },
                        "all_shapes": matches_found
                    }
            # END Gv14
            # --- ACTION: RETRIEVE (Flattened & Aligned) ---
            # BEGIN Gv13b
            elif action == 'retrieve':
                if not artifact_id:
                    return "Error: ID required."
                with h5py.File(active_vault_path, 'r') as f:
                    if artifact_id not in f:
                        return f"Error: '{artifact_id}' not found."

                    target = f[artifact_id]
                    stored_meta = dict(target.attrs)

                    # --- HELPER: Safe Retrieval Function ---
                    def safe_get(ds):
                        if ds.ndim == 0:
                            val = ds[()]
                            if isinstance(val, bytes):
                                val = val.decode('utf-8')

                            # PRODUCTION LOGIC: Auto-parse stringified dicts
                            if isinstance(val, str) and val.startswith('{') and val.endswith('}'):
                                try:
                                    import ast
                                    # We do the work here so the agent receives a clean DICT
                                    return ast.literal_eval(val)
                                except Exception as e:
                                    logging.warning(e)
                                    return val
                            return val
                        return ds[:]

                    # CASE A: Direct Dataset Access
                    if isinstance(target, h5py.Dataset):
                        return {
                            "samples": safe_get(target),
                            "metadata": stored_meta,
                            "vault_id": artifact_id,
                            "technical_bridge": "DIRECT_DATASET: Scalar-safe retrieval."
                        }

                    # CASE B: Group Access (Flattening)
                    '''
                    response = {"metadata": stored_meta, "vault_id": artifact_id}

                    keys = list(target.keys())
                    for k in keys:
                        if isinstance(target[k], h5py.Dataset):
                            # USE THE SAFE GETTER HERE TOO
                            response[k] = safe_get(target[k])

                    # BEGIN Gv16
                    # Inside Gv16 retrieve/inspect logic:
                    for k, v in stored_meta.items():
                        response[k] = v # Promote 'timestamp', 'window_index' to top level
                    # END Gv16
                    # Protocol Unification V3
                    for primary in ['samples', 'audio_samples', 'original_signal', 'raw_data', keys[0]]:
                        if primary in response:
                            response["samples"] = response[primary]
                            break

                    return response
                    '''

                    # CASE B: Group Access (Flattening)
                    # We start with a base response
                    response = {
                        "metadata": stored_meta,  # Keep the dict for structure
                        "vault_id": artifact_id
                    }

                    # --- Gv16: METADATA PROMOTION ---
                    # Promote attributes (timestamp, window_index, Fs) to top level
                    for k, v in stored_meta.items():
                        response[k] = v

                    # --- DATASET PROMOTION (DO NOT COMMENT THIS OUT!) ---
                    # Pull the actual arrays (imf_matrix, original_signal) to top level
                    keys = list(target.keys())
                    for k in keys:
                        if isinstance(target[k], h5py.Dataset):
                            response[k] = safe_get(target[k])

                    # --- PROTOCOL UNIFICATION V3 ---
                    # Now that everything is at the top level, find the best 'samples' alias
                    for primary in ['samples', 'audio_samples', 'original_signal', 'raw_data']:
                        if primary in response:
                            response["samples"] = response[primary]
                            break

                    # Absolute Fallback: if no primary key found, use the first available dataset
                    if "samples" not in response and keys:
                        response["samples"] = response[keys[0]]

                    return response

            # END Gv13b
            # --- ACTION: STASH ---
            elif action == 'stash':
                if not artifact_id or data is None:
                    return "Error: ID and data required."

                with h5py.File(active_vault_path, 'a') as f:  # <--- Use local path
                    if artifact_id in f:
                        del f[artifact_id]
                    group = f.create_group(artifact_id)

                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, np.ndarray):
                                group.create_dataset(key,
                                                     data=value,
                                                     compression="gzip",
                                                     compression_opts=4)
                            else:
                                group.create_dataset(key, data=str(value))
                    elif isinstance(data, np.ndarray):
                        group.create_dataset('raw_data',
                                             data=data,
                                             compression="gzip",
                                             compression_opts=4)
                    else:
                        group.create_dataset('raw_data', data=str(data))

                    group.attrs['timestamp'] = str(datetime.now())
                    group.attrs['user_instruction'] = str(access_hint) if access_hint else "N/A"
                    if metadata:
                        for k, v in metadata.items():
                            group.attrs[k] = v

                    return {"status": "SUCCESS",
                            "passport": self._generate_passport(artifact_id,
                                                                data,
                                                                access_hint
                                                                )
                            }

        except Exception as e:
            return f"Vault error: {str(e)}"

    def _generate_passport(self, id, data, hint):
        p = {"id": id, "instructional_manual": hint if hint else "N/A", "timestamp": str(datetime.now())}
        if hasattr(data, 'shape'):
            p["shape"] = list(data.shape)
        return p