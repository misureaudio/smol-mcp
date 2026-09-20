class LLM_Tool(SafeOpenTool):
    name = "llm_tool"
    description = ("This is a tool that accepts a question (and possibly a list of images) and returns an answer.")
    inputs = {
        "question":
        {
            "type": "string",
            "description": "The question to be answered. If it contains web URLs pointing to images, they must be explored with this tool, rather than using visit_webpage()."
        },
        "list_of_images":
        {
            "type": "object",
            "description": "An optional list, possibly empty, containing a mix of base64-encoded .JPG or .PNG images, paths to local image files, and open file objects containing images.",
            "nullable": True
        }
    }
    output_type = "string"

    # MODIFICATO: L'init accetta un dizionario 'config' e 'safe_dir'
    def __init__(self, config: dict, safe_dir: str):
        # Chiamata corretta al costruttore della superclasse
        super(LLM_Tool, self).__init__(safe_dir)
        self.config = config
        self.safe_dir = safe_dir
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("Secure LLMTool initialized.")

    def forward(self, question, list_of_images=[]):
        print(f"*** In LLM_Tool.forward con configurazione: {self.config}", file=sys.stderr)

        content = [
            {"type": "text", "text": question}
        ]

        for img in list_of_images:
            invalid_argument = True
            if isinstance(img, str):
                if re.match(valid_url_pattern, img) is not None:
                    invalid_argument = False
                elif os.path.isfile(img):
                    img = super().forward(img, "rb")
                    invalid_argument = False
            if isinstance(img, Iterable) and not isinstance(img, (str, bytes)) and callable(getattr(img, 'read')):
                imgb64 = base64.b64encode(img.read()).decode('utf-8')
                img.close()
                img = imgb64
                invalid_argument = False
            if invalid_argument:
                raise TypeError(f"Invalid argument {img} (type: {type(img).__name__})")

            content += [{"type": "image_url", "image_url": {"url": img}}]

        # MODIFICATO: I parametri vengono letti dal dizionario self.config
        # Vengono usati valori di default se una chiave non è presente
        modloc = self.config.get("model", "ollama/gemma3:4b-it-qat")  # Default se non specificato
        numctx = self.config.get("context", 131072)
        timout = self.config.get("timeout", 57600)
        temloc = self.config.get("temperature", 0.7)

        pp(f"LLM Tool Model  : {modloc}")
        pp(f"LLM Tool num_ctx: {numctx}")
        pp(f"LLM Tool timeout: {timout}")
        pp(f"LLM Tool temperature: {temloc}")

        response = completion(
            model=modloc,
            api_base=f"{GB10HOST}",
            api_key=f"{GB10KEY}",
            messages=[{"role": "user", "content": content}],
            timeout=int(timout),
            num_ctx=int(numctx),
            temperature=float(temloc),
        )

        response_content = response.choices[0].message.content
        response_content = re.sub(ThinkStr,
                                  # r'(?:<think>.*?</think>|<thought>.*?</thought>|Thinking....*?...done thinking|\[THINK\].*?\[/THINK\]|.*?</think>|.*?</thought>|.*?...done thinking)|.*?\[/THINK\](.*)',
                                  '', response_content, flags=re.DOTALL)
        return response_content