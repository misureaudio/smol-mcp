class ManageChromaDBTool(Tool):
    name = "manage_chromadb"
    # ... (description and inputs remain the same) ...
    description = (
        "Manages a ChromaDB database with query and delete functions. "
        "Query actions: 'list_collections', 'get_all_memories', 'search'. "
        "Delete actions: 'delete_collection', 'delete_memories_by_ids', 'delete_memories_by_metadata'."
    )
    # ... (inputs remain the same) ...
    inputs = {
        "db_path": {
            "type": "string",
            "description": "Path to the ChromaDB database to manage."
        },
        "action": {
            "type": "string",
            "description": "The action to perform. See tool description for supported actions."
        },
        "collection_name": {
            "type": "string",
            "description": "The target collection name. Required for most actions.",
            "nullable": True
        },
        "query_text": {
            "type": "string",
            "description": "The search query text for the 'search' action.",
            "nullable": True
        },
        "n_results": {
            "type": "integer",
            "description": "Number of results for 'search' action. Defaults to 5.",
            "nullable": True
        },
        "memory_ids": {
            "type": "object",
            "description": "A list of memory string IDs to delete. Required for 'delete_memories_by_ids'.",
            "nullable": True
        },
        "metadata_filter": {
            "type": "object",
            "description": "A dictionary for metadata filtering (e.g., {'user_id': 'alice'}). Required for 'delete_memories_by_metadata'.",
            "nullable": True
        },
        "embedder_model": {
            "type": "string",
            "description": "The sentence-transformer model for search query embedding. Defaults to 'sentence-transformers/distiluse-base-multilingual-cased-v2'.",
            "nullable": True
        }
    }
    output_type = "object"

    def __init__(self, config: dict):
        super().__init__()
        # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.config = config
        logging.info("ManageChromaDBTool initialized.")

    # ... (_get_memory_client method remains the same) ...
    def _get_memory_client(self, db_path, collection_name, embedder_model):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at: {db_path}")

        # MODIFICATO: I parametri vengono letti dal dizionario self.config
        # Vengono usati valori di default se una chiave non è presente
        model4mem0 = self.config.get("model")  # Default se non specificato
        # numctx = self.config.get("context")
        # timout = self.config.get("timeout", 57600)
        temp4mem0 = self.config.get("temperature")

        pp(f"mem0 Tool Model  : {model4mem0}")
        # pp(f"mem0 Tool num_ctx: {numctx}")
        # pp(f"mem0 Tool timeout: {timout}")
        pp(f"mem0 Tool temperature: {temp4mem0}")

        configur = {
            "embedder": {"provider": "huggingface", "config": {"model": embedder_model}},
            "vector_store": {"provider": "chroma", "config": {"collection_name": collection_name or "default_collection", "path": db_path}},
            "llm": {"provider": "ollama",
                        "ollama_base_url": f"{GB10HOST}",
                        "api_key": f"{GB10KEY}",
                        "config": {"model": model4mem0,
                                   "temperature": temp4mem0}}
        }
        return SharedMem0Manager.get_memory(configur)

    # --- FIX APPLIED HERE ---
    def forward(self, db_path, action, collection_name=None, query_text=None, n_results=None, memory_ids=None, metadata_filter=None, embedder_model=None):
        if n_results is None:
            n_results = 5
        if embedder_model is None:
            embedder_model = "sentence-transformers/distiluse-base-multilingual-cased-v2"

        try:
            # ... (rest of the method remains the same) ...
            # Query Actions
            if action == "list_collections":
                memory_instance = self._get_memory_client(db_path,
                                                          "any_collection_for_client",
                                                          embedder_model)
                client = memory_instance.vector_store.client
                return json.dumps({"collections": [col.name for col in client.list_collections()]})

            elif action == "get_all_memories":
                if not collection_name:
                    return "Error: 'collection_name' is required."
                memory_client = self._get_memory_client(db_path,
                                                        collection_name,
                                                        embedder_model)
                return json.dumps(memory_client.vector_store.collection.get(include=["documents", "metadatas"]), indent=2)

            elif action == "search":
                if not collection_name or not query_text:
                    return "Error: 'collection_name' and 'query_text' are required."
                memory_client = self._get_memory_client(db_path,
                                                        collection_name,
                                                        embedder_model)

                # --- CORRECTION: Use ChromaDB's direct query method ---
                embedder = SentenceTransformer(embedder_model)
                query_embedding = embedder.encode([query_text])
                collection = memory_client.vector_store.collection
                results = collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                # return json.dumps(results, indent=2)
                return results

            # Delete Actions
            elif action == "delete_collection":
                if not collection_name:
                    return "Error: 'collection_name' is required."
                memory_instance = self._get_memory_client(db_path, collection_name, embedder_model)
                client = memory_instance.vector_store.client
                client.delete_collection(name=collection_name)
                SharedMem0Manager.clear_instance(db_path, collection_name)
                return f"Successfully deleted collection: {collection_name}"

            elif action == "delete_memories_by_ids":
                if not collection_name or not memory_ids:
                    return "Error: 'collection_name' and 'memory_ids' are required."
                memory_client = self._get_memory_client(db_path,
                                                        collection_name,
                                                        embedder_model)
                memory_client.vector_store.collection.delete(ids=memory_ids)
                return f"Successfully deleted {len(memory_ids)} memories from collection: {collection_name}"

            elif action == "delete_memories_by_metadata":
                if not collection_name or not metadata_filter:
                    return "Error: 'collection_name' and 'metadata_filter' are required."
                memory_client = self._get_memory_client(db_path,
                                                        collection_name,
                                                        embedder_model)
                collection = memory_client.vector_store.collection
                ids_to_delete = collection.get(where=metadata_filter, include=[]).get("ids", [])
                if not ids_to_delete:
                    return f"No memories found matching the filter in collection: {collection_name}"
                collection.delete(where=metadata_filter)
                return f"Successfully deleted {len(ids_to_delete)} memories from collection: {collection_name}"

            else:
                return f"Error: Invalid action '{action}'."
        except Exception as e:
            logging.exception("An exception occurred in ManageChromaDBTool")
            error_message = f"An error occurred during management operation: {e}"
            return f"Error: {error_message}"