class PopulateChromaDBTool(SafeOpenTool):
    name = "populate_chromadb"
    # ... (description and inputs remain the same) ...
    description = (
        "Reads a markdown dialog file, extracts memories using an LLM, and populates "
        "a specified ChromaDB collection with those memories."
    )
    inputs = {
        "db_path": {
            "type": "string",
            "description": "Path to the ChromaDB database to populate."
        },
        "markdown_file": {
            "type": "string",
            "description": "Path to the markdown dialog file containing the conversation."
        },
        "collection_name": {
            "type": "string",
            "description": "The name of the collection to add the memories to."
        },
        "user_id": {
            "type": "string",
            "description": "A user ID to be stored in the metadata for each memory."
        },
        "embedder_model": {
            "type": "string",
            "description": "The sentence-transformer model to use for embedding. Defaults to 'sentence-transformers/distiluse-base-multilingual-cased-v2'.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, config: dict, safe_dir):
        super().__init__(safe_dir)
        # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.config = config
        logging.info("PopulateChromaDBTool initialized.")
        self.SYSTEM_PROMPT = """
You are a memory management system. Your purpose is to process conversation history and store key information as memories.
You must analyze the user's query and respond ONLY with a JSON object containing a list of memories.
Each memory in the list should be a JSON object with "id", "text", and "event" keys.
The only possible event type is "INSIGHT".
Extract meaningful insights, preferences, facts, and important information from the conversation.
Example:
{"memories": [{"id": 0, "text": "The user's name is John.", "event": "INSIGHT"}]}
"""

    # ... (_read_markdown_dialog and _extract_memories_from_dialog methods remain the same) ...
    def _read_markdown_dialog(self, file_path):
        try:
            file = super().forward(file_path, "rt")
            return file.read()
        except FileNotFoundError:
            raise ValueError(f"Markdown file not found: {file_path}")
        except Exception as e:
            raise IOError(f"Error reading markdown file: {e}")

    def _extract_memories_from_dialog(self, mem0_agent, dialog_content):
        # print(self.SYSTEM_PROMPT)
        # print(dialog_content)
        try:
            messages_for_llm = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Please analyze the following conversation and extract memories:\n\n{dialog_content}"}
            ]
            llm_response_str = mem0_agent.llm.generate_response(
                                            messages=messages_for_llm)

            # Parse the LLM's response, handling Chain-of-Thought tags
            match = re.search(ThinkStr,
                              llm_response_str,
                              re.DOTALL
                              )
            result = None
            if match:
                # group(1) contains the content captured by the parentheses (.*)
                result = match.group(1).strip()  # .strip() removes leading/trailing whitespace
                print(f"CoT Filtered query : '{result}'")
                llm_response_str = result
                print("✓ Thinking tags stripped: " + llm_response_str)
            else:
                print("✓ LLM responded successfully: " + llm_response_str)

            print("Parsing LLM response to extract clean JSON...")
            json_start_index = llm_response_str.find('{')
            if json_start_index == -1:
                raise ValueError("✗ No JSON object found in the LLM's response")

            json_str = llm_response_str[json_start_index:]
            extracted_data = extract_json_robust(json_str)
            memories = extracted_data.get("memories", [])
            print(f"✓ Successfully extracted {len(memories)} memories from dialog")
            return memories

        except (json.JSONDecodeError, ValueError) as e:
            print(f"✗ Error parsing LLM response: {e}")
            return []
        except Exception as e:
            print(f"✗ Error extracting memories from LLM: {e}")
            return []

    # --- FIX APPLIED HERE ---

    def forward(self, db_path, markdown_file, collection_name, user_id, embedder_model=None):
        if embedder_model is None:
            embedder_model = "sentence-transformers/distiluse-base-multilingual-cased-v2"

        try:
            # ... (rest of the method remains the same) ...
            if not os.path.exists(db_path):
                return f"Error: Database not found at '{db_path}'. Please create it first."

            dialog_content = self._read_markdown_dialog(markdown_file)
            if not dialog_content:
                return "Error: Markdown file is empty or could not be read."

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
                "embedder": {"provider": "huggingface",
                             "config": {"model": embedder_model}},
                "vector_store": {"provider": "chroma",
                                 "config": {"collection_name": collection_name, "path": db_path}},
                "llm": {"provider": "ollama",
                        "ollama_base_url": f"{GB10HOST}",
                        "api_key": f"{GB10KEY}",
                        "config": {"model": model4mem0,
                                   "temperature": temp4mem0}}
            }

            # MODIFICA PRINCIPALE: Usa SharedMem0Manager invece di Memory.from_config
            mem0_agent = SharedMem0Manager.get_memory(configur)

            memories = self._extract_memories_from_dialog(mem0_agent,
                                                          dialog_content)
            if not memories:
                return "No memories were extracted from the dialog."

            embedder = SentenceTransformer(embedder_model)
            docs_to_embed = [memory["text"] for memory in memories]
            embeddings = embedder.encode(docs_to_embed, show_progress_bar=True)

            timestamp = datetime.now().isoformat()
            ids_to_add = [str(uuid.uuid4()) for _ in memories]
            metadatas_to_add = [{
                "user_id": user_id,
                "collection_name": collection_name,
                "timestamp": timestamp,
                "source": "markdown_dialog",
                "memory_type": memory.get("event", "INSIGHT")
            } for memory in memories]

            mem0_agent.vector_store.collection.add(
                ids=ids_to_add,
                documents=docs_to_embed,
                embeddings=embeddings.tolist(),
                metadatas=metadatas_to_add
            )

            return f"Successfully populated collection '{collection_name}' with {len(memories)} memories for user '{user_id}'."
        except Exception as e:
            error_message = f"An error occurred while populating the database: {e}"
            logging.error(error_message)
            return f"Error: {error_message}"