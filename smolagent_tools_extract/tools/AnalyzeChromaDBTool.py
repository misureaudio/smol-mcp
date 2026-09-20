class AnalyzeChromaDBTool(Tool):
    """
    A tool to find and verify mem0 ChromaDB structures on the local filesystem.
    """
    name = "analyze_chromadb"
    description = (
        "Finds and verifies mem0 ChromaDB structures. "
        "Actions: 'find' to search for potential databases in a path, "
        "'verify' to check if a specific path is a valid ChromaDB structure."
    )
    inputs = {
        "action": {
            "type": "string",
            "description": "The action to perform: 'find' or 'verify'."
        },
        "search_path": {
            "type": "string",
            "description": "The directory path to start the search from (for 'find' action).",
            "nullable": True
        },
        "db_path": {
            "type": "string",
            "description": "The path to a potential ChromaDB directory to verify (for 'verify' action).",
            "nullable": True
        }
    }
    output_type = "object"

    def _is_uuid_like(self, name: str) -> bool:
        """
        Checks if a string matches the 8-4-4-4-12 UUID pattern.
        """
        uuid_pattern = re.compile(
            r'^[0-9a-fA-F]{8}-'
            r'[0-9a-fA-F]{4}-'
            r'[0-9a-fA-F]{4}-'
            r'[0-9a-fA-F]{4}-'
            r'[0-9a-fA-F]{12}$'
        )
        return bool(uuid_pattern.match(name))

    def _find_potential_mem0_dbs(self, search_path: str) -> list:
        """
        Finds potential mem0 ChromaDB structures within a given path.
        """
        potential_dbs = []
        for root, _, files in os.walk(search_path):
            if "chroma.sqlite3" in files:
                has_uuid_sibling = False
                for d in os.listdir(root):
                    if os.path.isdir(os.path.join(root, d)) and self._is_uuid_like(d):
                        has_uuid_sibling = True
                        break
                if has_uuid_sibling:
                    potential_dbs.append(root)
        return potential_dbs

    def _is_real_mem0_data(self, db_path: str) -> bool:
        """
        Analyzes a potential ChromaDB path to determine if it's a real mem0 data store.
        This confirms the structure, but not the content.
        """
        if not os.path.isdir(db_path):
            return False

        sqlite_file = os.path.join(db_path, "chroma.sqlite3")
        if not os.path.isfile(sqlite_file):
            return False

        has_uuid_dir = False
        for item in os.listdir(db_path):
            item_path = os.path.join(db_path, item)
            if os.path.isdir(item_path) and self._is_uuid_like(item):
                has_uuid_dir = True
                break
        return has_uuid_dir

    # -'-'-'
    def __init__(self):
        super().__init__()
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("AnalyzeChromaDBTool initialized.")
    # -'-'-'

    def forward(self, action: str, search_path: str = None, db_path: str = None):
        if action == "find":
            if not search_path:
                return "Error: 'search_path' is required for 'find' action."
            if not os.path.isdir(search_path):
                return f"Error: The search path '{search_path}' does not exist or is not a directory."
            found_dbs = self._find_potential_mem0_dbs(search_path)
            if not found_dbs:
                return f"No potential mem0 ChromaDBs found in '{search_path}'."
            return {
                "status": "success",
                "found_databases": found_dbs
            }

        elif action == "verify":
            if not db_path:
                return "Error: 'db_path' is required for 'verify' action."
            is_real = self._is_real_mem0_data(db_path)
            return {
                "db_path": db_path,
                "is_real_mem0_chromadb_structure": is_real
            }

        else:
            return f"Error: Invalid action '{action}'. Supported actions are 'find' and 'verify'."