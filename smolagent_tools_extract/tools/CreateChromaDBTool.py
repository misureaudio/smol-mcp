class CreateChromaDBTool(Tool):
    name = "create_chromadb"
    description = (
        "Creates a new, empty ChromaDB vector database at a specified path. "
        "If a database already exists at the path, it will be removed and replaced."
    )
    inputs = {
        "db_path": {
            "type": "string",
            "description": "The file system path where the ChromaDB database will be created. Defaults to './chroma_db_mem0'.",
            "nullable": True
        },
        "embedder_model": {
            "type": "string",
            "description": "The sentence-transformer model to configure for the database. Defaults to 'sentence-transformers/distiluse-base-multilingual-cased-v2'.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, config: dict):
        super().__init__()
        # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.config = config
        logging.info("CreateChromaDBTool initialized.")

    # --- FIX APPLIED HERE ---
    # Removed all type hints from the signature and kept default None values.
    def forward(self, db_path=None, embedder_model=None):
        """
        Creates an empty ChromaDB database with the specified configuration.
        """
        # Set default values if arguments are not provided
        if db_path is None:
            db_path = "./chroma_db_mem0"
        if embedder_model is None:
            embedder_model = "sentence-transformers/distiluse-base-multilingual-cased-v2"

        try:
            if os.path.exists(db_path):
                logging.info(f"Removing existing '{db_path}' directory for a clean setup...")
                shutil.rmtree(db_path)
                SharedMem0Manager.clear_all()

            model4mem0 = self.config.get("model")
            temp4mem0 = self.config.get("temperature")

            pp(f"mem0 Tool Model  : {model4mem0}")
            pp(f"mem0 Tool temperature: {temp4mem0}")

            configur = {
                "embedder": {"provider": "huggingface", "config": {"model": embedder_model}},
                "vector_store": {"provider": "chroma", "config": {"collection_name": "default_collection", "path": db_path}},
                "llm": {"provider": "ollama",
                        "ollama_base_url": f"{GB10HOST}",
                        "api_key": f"{GB10KEY}",
                        "config": {"model": model4mem0,
                                   "temperature": temp4mem0}}
            }

            logging.info(f"Creating empty ChromaDB at: {db_path} with embedder: {embedder_model}")

            memory_instance = SharedMem0Manager.get_memory(configur)
            SharedMem0Manager.clear_instance(db_path, "default_collection")

            if os.path.exists(db_path):
                success_message = f"Successfully created ChromaDB at: {db_path}"
                logging.info(success_message)
                return success_message
            else:
                error_message = "Failed to create ChromaDB directory."
                logging.error(error_message)
                return f"Error: {error_message}"

        except Exception as e:
            error_message = f"An error occurred while creating ChromaDB: {e}"
            logging.error(error_message)
            return f"Error: {error_message}"