class QueryChromaDBTool(Tool):
    name = "query_chromadb"
    # ... (description and inputs remain the same) ...
    description = (
        "Queries a ChromaDB database. It can perform several actions: "
        "'list_collections': Lists all available collections. "
        "'get_all_memories': Retrieves all memories from a specific collection. "
        "'search': Performs a similarity search for a query within a collection."
    )
    inputs = {
        "db_path": {
            "type": "string",
            "description": "Path to the ChromaDB database to query."
        },
        "action": {
            "type": "string",
            "description": "The action to perform. Supported values: 'list_collections', 'get_all_memories', 'search'."
        },
        "collection_name": {
            "type": "string",
            "description": "The name of the collection for 'get_all_memories' and 'search' actions.",
            "nullable": True
        },
        "query_text": {
            "type": "string",
            "description": "The search query text for the 'search' action.",
            "nullable": True
        },
        "n_results": {
            "type": "integer",
            "description": "The number of results to return for a 'search' action. Defaults to 5.",
            "nullable": True
        },
        "embedder_model": {
            "type": "string",
            "description": "The sentence-transformer model for embedding the search query. Defaults to 'sentence-transformers/distiluse-base-multilingual-cased-v2'.",
            "nullable": True
        }
    }
    output_type = "object"

    def __init__(self, config: dict):
        super().__init__()
        # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.config = config
        logging.info("QueryChromaDBTool initialized.")

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
            "embedder": {
                "provider": "huggingface",
                "config": {"model": embedder_model}},
            "vector_store": {"provider": "chroma",
                             "config": {"collection_name": collection_name or "default_collection",
                                        "path": db_path}},
            "llm": {"provider": "ollama",
                        "ollama_base_url": f"{GB10HOST}",
                        "api_key": f"{GB10KEY}",
                        "config": {"model": model4mem0,
                                   "temperature": temp4mem0}}
        }
        return SharedMem0Manager.get_memory(configur)

    # --- FIX APPLIED HERE ---
    def forward(self, db_path, action, collection_name=None, query_text=None, n_results=None, embedder_model=None):
        if n_results is None:
            n_results = 5
        if embedder_model is None:
            embedder_model = "sentence-transformers/distiluse-base-multilingual-cased-v2"

        try:
            # ... (rest of the method remains the same) ...
            if action == "list_collections":
                memory_instance = self._get_memory_client(db_path,
                                                          "any_collection_for_client",
                                                          embedder_model)
                client = memory_instance.vector_store.client
                collections = client.list_collections()
                collection_names = [col.name for col in collections]
                return json.dumps({"collections": collection_names})

            elif action == "get_all_memories":
                if not collection_name:
                    return "Error: 'collection_name' is required for 'get_all_memories' action."

                memory_client = self._get_memory_client(db_path,
                                                        collection_name,
                                                        embedder_model)
                results = memory_client.vector_store.collection.get(include=["documents", "metadatas"])
                # return json.dumps(results, indent=2)
                return results

            elif action == "search":
                if not collection_name or not query_text:
                    return "Error: 'collection_name' and 'query_text' are required for 'search' action."

                memory_client = self._get_memory_client(db_path,
                                                        collection_name,
                                                        embedder_model)

                # --- CORRECTION: Use ChromaDB's direct query method for general search ---
                # Initialize the embedder to convert the query text into a vector
                embedder = SentenceTransformer(embedder_model)
                query_embedding = embedder.encode([query_text])

                # Access the underlying ChromaDB collection via the mem0 instance
                collection = memory_client.vector_store.collection

                # Execute the vector similarity query
                results = collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                # return json.dumps(results, indent=2)
                return results

            else:
                return f"Error: Invalid action '{action}'. Supported actions are: 'list_collections', 'get_all_memories', 'search'."
        except Exception as e:
            logging.exception("An exception occurred in QueryChromaDBTool")
            error_message = f"An error occurred during query: {e}"
            return f"Error: {error_message}"