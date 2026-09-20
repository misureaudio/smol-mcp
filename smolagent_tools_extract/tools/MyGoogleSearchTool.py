class MyGoogleSearchTool(Tool):
    name = "web_search"
    description = """Performs a web search based on your query (think a Google search) then returns the top search results."""
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results=10, engine=""):
        super().__init__()
        self.max_results = max_results
        self.engine = engine  # 'google' or 'duckduckgo'
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("MyGoogleSearchTool initialized.")

        if self.engine == "google":
            try:
                from googlesearch import search as gsearch
            except ImportError as e:
                raise ImportError(
                    ("You must install the package `googlesearch` to run this tool: "
                     "for instance run `pip install googlesearch-python`.")
                ) from e
            self.gsearch = gsearch
        elif self.engine == "duckduckgo":
            try:
                from ddgs import DDGS
            except ImportError as e:
                raise ImportError(
                    ("You must install the package `duckduckgo-search` to run this tool: "
                     "for instance run `pip install duckduckgo-search`.")
                ) from e
            # self.ddgs = DDGS()
            # BEGIN ENZO: si assegna solo reference in __init__()
            self.DDGS = DDGS
            # END ENZO
        else:
            raise ValueError("Engine must be either 'google' or 'duckduckgo'")

    def forward(self, query: str) -> str:
        if self.engine == "google":
            results = list(self.gsearch(query, num_results=self.max_results,
                                        unique=True, advanced=True))
            if len(results) == 0:
                raise Exception("No results found! Try a less restrictive/shorter query.")
            postprocessed_results = [f"[{result.title}]({result.url})\n{result.description}" for result in results]
        elif self.engine == "duckduckgo":
            try:
                # results = list(self.ddgs.text(query, max_results=self.max_results))
                # BEGIN ENZO: la classe DDGS viene istanziata nella forward per garantire che più query siano supportate ciascuna dalla propria istanza
                ddgs = self.DDGS()
                results = list(ddgs.text(query, max_results=self.max_results))
                # END ENZO
                if not results:
                    raise Exception("No results found! Try a less restrictive/shorter query.")
                postprocessed_results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
            except Exception as e:
                raise Exception(f"DuckDuckGo search failed: {str(e)}")
            # Il gc dovrebbe distruggere la istanza DDG dopo l'uso
            ddgs = None

        # returns a list of ~10 strings formatted as [title](URL)\ndescription
        return "## Search Results\n\n" + "\n\n".join(postprocessed_results)