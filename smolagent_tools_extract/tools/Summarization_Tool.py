class Summarization_Tool(LLM_Tool):
    name = "summarization_tool"
    description = ("This is a tool that summarizes the input (a long multi-line string)")
    inputs = {
        "long_text": {"type": "string", "description": "The text to be summarized."},
        "list_of_images": {"type": "object", "description": "An optional list...", "nullable": True}
    }
    output_type = "string"

    # MODIFICATO: L'init deve corrispondere a quello della classe base (LLM_Tool)
    def __init__(self, config: dict, safe_dir: str):
        super().__init__(config, safe_dir)  # Passa i parametri al costruttore di LLM_Tool
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("Secure Summarization_Tool initialized.")

    def forward(self, long_text, list_of_images=[]):
        # La chiamata a super().forward utilizzerà automaticamente la configurazione corretta
        return super().forward(r'summarize the following text:\n\n' + long_text, list_of_images)