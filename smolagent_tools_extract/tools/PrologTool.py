class PrologTool(SafeOpenTool):
    """
    A tool for interacting with a SWI-Prolog interpreter via pyswip.
    Supports symbolic reasoning, knowledge representation, constraint solving,
    and persistence (save/load KB to file, auto-save on close).
    File operations are restricted to a specified safe directory.

    Knowledge persists across calls by default, but you can reset dynamically:
      - reset_on_query=True → clear KB before each query
      - reset_on_consult=True → clear KB before consulting
      - reset_on_all=True → shorthand: clear KB before both consult and query

    Persistence:
      - save_to="file.pl" → save KB to a file in the safe directory
      - load_from="file.pl" → load KB from a file in the safe directory
      - auto_save_file="file.pl" → automatically save KB on close to the safe directory (default: session.pl)
    """
    name = "prolog_tool"
    description = (
        "Interface to SWI-Prolog. Supports consulting facts/rules, running queries, "
        "clearing the knowledge base, saving/loading KB to/from files, "
        "and auto-saving on close. File operations are restricted to a safe directory."
    )
    inputs = {
        "action": {
            "type": "string",
            "description": "Operation: 'consult', 'query', 'clear', 'save', 'load'."
        },
        "program": {
            "type": "string",
            "description": "Prolog clauses (facts/rules). Required for 'consult'.",
            "nullable": True
        },
        "query_string": {
            "type": "string",
            "description": "Prolog query to execute. Required for 'query'.",
            "nullable": True
        },
        "find_all": {
            "type": "boolean",
            "description": "For 'query': return all solutions (True) or just the first (False).",
            "nullable": True
        },
        "reset_on_query": {
            "type": "boolean",
            "description": "If True, resets the KB before each query.",
            "nullable": True
        },
        "reset_on_consult": {
            "type": "boolean",
            "description": "If True, resets the KB before consulting new facts/rules.",
            "nullable": True
        },
        "reset_on_all": {
            "type": "boolean",
            "description": "If True, resets the KB before BOTH consult and query.",
            "nullable": True
        },
        "save_to": {
            "type": "string",
            "description": "Path to save the current KB (Prolog .pl file). Will be saved inside the safe directory.",
            "nullable": True
        },
        "load_from": {
            "type": "string",
            "description": "Path to load a KB from a Prolog .pl file. Must exist inside the safe directory.",
            "nullable": True
        }
    }
    output_type = "object"

    def __init__(self, safe_dir: str, reset_on_query: bool = False, reset_on_consult: bool = False,
                 reset_on_all: bool = False, auto_save_file: str = "session.pl"):
        super().__init__(safe_dir=safe_dir)
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info(f"PrologTool initialized.\n*** ✅ ---> Prolog dbs will be saved in '{os.path.abspath(self.safe_dir)}'")
        try:
            self.prolog = Prolog()
            self.reset_on_query_default = reset_on_query
            self.reset_on_consult_default = reset_on_consult
            self.reset_on_all_default = reset_on_all
            self.auto_save_file = auto_save_file
            logging.info("SWI-Prolog instance initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize SWI-Prolog. Ensure SWI-Prolog is installed and in PATH. Error: {e}")
            self.prolog = None

    def _cleanup_program_string(self, program: str) -> list[str]:
        """Split a program string into individual clauses ending with '.'"""
        clauses = re.split(r"\.\s*(?=\w|\()", program.strip())
        return [clause.strip() + "." for clause in clauses if clause.strip()]

    def forward(self, action: str, program: str = None, query_string: str = None,
                find_all: bool = True, reset_on_query: bool = None, reset_on_consult: bool = None,
                reset_on_all: bool = None, save_to: str = None, load_from: str = None):
        if not self.prolog:
            return "Error: Prolog interpreter is not available."

        # Resolve reset flags
        if reset_on_query is None:
            reset_on_query = self.reset_on_query_default
        if reset_on_consult is None:
            reset_on_consult = self.reset_on_consult_default
        if reset_on_all is None:
            reset_on_all = self.reset_on_all_default
        if reset_on_all:
            reset_on_query = True
            reset_on_consult = True

        if action == "consult":
            if not program:
                return "Error: 'program' is required for 'consult'."
            try:
                if reset_on_consult:
                    self.prolog = Prolog()
                clauses = self._cleanup_program_string(program)
                for clause in clauses:
                    self.prolog.assertz(clause.rstrip("."))
                return f"Consulted {len(clauses)} clauses."
            except PrologError as e:
                return f"Error consulting program: {e}"

        elif action == "query":
            if not query_string:
                return "Error: 'query_string' is required for 'query'."
            try:
                if reset_on_query:
                    self.prolog = Prolog()
                q_str = query_string.strip()
                if not q_str.endswith("."):
                    q_str += "."
                q = self.prolog.query(q_str)
                if find_all:
                    results = list(q)
                    q.close()
                    if not results:
                        return "Query returned no results (false)."
                    if all(r == {} for r in results):
                        return "Query was successful (true)."
                    return results
                else:
                    result = next(q, None)
                    q.close()
                    if result is None:
                        return "Query returned no results (false)."
                    if result == {}:
                        return "Query was successful (true)."
                    return result
            except PrologError as e:
                return f"Error executing query: {e}"

        elif action == "clear":
            self.prolog = Prolog()
            return "Knowledge base cleared."

        elif action == "save":
            if not save_to:
                return "Error: 'save_to' file path is required."

            # Security: construct a safe path inside the allowed directory
            safe_save_path = os.path.join(self.safe_dir, os.path.basename(save_to))

            # --- FIX: NORMALIZE PATH SEPARATORS FOR PROLOG ---
            prolog_safe_path = safe_save_path.replace('\\', '/')

            try:
                # Use the normalized path in the query
                list(self.prolog.query(f"tell('{prolog_safe_path}')"))
                list(self.prolog.query("listing."))
                list(self.prolog.query("told."))
                return f"Knowledge base saved to {safe_save_path}."
            except Exception as e:
                return f"Error saving KB: {e}"

        elif action == "load":
            if not load_from:
                return "Error: 'load_from' file path is required."

            # Security: construct a safe path inside the allowed directory
            safe_load_path = os.path.join(self.safe_dir, os.path.basename(load_from))

            # --- FIX: NORMALIZE PATH SEPARATORS FOR PROLOG ---
            prolog_safe_path = safe_load_path.replace('\\', '/')

            if not os.path.exists(safe_load_path):
                return f"Error: file {safe_load_path} not found in the safe directory."
            try:
                # Use the normalized path
                self.prolog.consult(prolog_safe_path)
                return f"Knowledge base loaded from {safe_load_path}."
            except Exception as e:
                return f"Error loading KB: {e}"

        else:
            return f"Error: Invalid action '{action}'. Supported: consult, query, clear, save, load."

    def close(self):
        """Save KB automatically and shut down Prolog."""
        if self.prolog:
            try:
                # Auto-save knowledge base before closing
                if self.auto_save_file:
                    # Security: ensure auto-save happens in the safe directory
                    safe_auto_save_path = os.path.join(self.safe_dir, os.path.basename(self.auto_save_file))

                    # --- FIX: NORMALIZE PATH SEPARATORS FOR PROLOG ---
                    prolog_safe_path = safe_auto_save_path.replace('\\', '/')

                    logging.info(f"Auto-saving KB to {safe_auto_save_path}...")
                    list(self.prolog.query(f"tell('{prolog_safe_path}')"))
                    list(self.prolog.query("listing."))
                    list(self.prolog.query("told."))

                logging.info("Halting SWI-Prolog...")
                next(self.prolog.query("halt."), None)
            except Exception as e:
                logging.info(f"Prolog halted (expected): {e}")
            finally:
                self.prolog = None
                logging.info("Prolog instance shut down.")