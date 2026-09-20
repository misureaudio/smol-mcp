class Translation_Tool(LLM_Tool):
    name = "translation_tool"
    description = ("This is a tool that translates the input (a long multi-line string) from one language to another.")
    inputs = {
        "original_language": {"type": "string", "description": "The original language of the text."},
        "target_language": {"type": "string", "description": "The target language for the translation."},
        "long_text": {"type": "string", "description": "The text to be summarized."},
        "list_of_images": {"type": "object", "description": "An optional list...", "nullable": True}
    }
    output_type = "string"

    # MODIFICATO: L'init deve corrispondere a quello della classe base (LLM_Tool)
    def __init__(self, config: dict, safe_dir: str):
        super().__init__(config, safe_dir)  # Passa i parametri al costruttore di LLM_Tool
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("Secure Translation_Tool initialized.")

    def forward(self, original_language, target_language, long_text, list_of_images=[]):
        # La chiamata a super().forward utilizzerà automaticamente la configurazione corretta
        return super().forward(r'translate the following text:\n\n from ' + original_language + ' to ' + target_language + ':\n\n' + long_text, list_of_images)