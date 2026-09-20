class SearXNGSearchTool(Tool):
    name = "web_searXNG"
    description = """Performs a web search based on your query (think a Google search) then returns the top search results."""
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results=10):
        super().__init__()
        self.max_results = max_results
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("SearXNGSearchTool initialized.")

    def forward(self, query: str) -> str:
        results = xngsearch(query, self.max_results)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        postprocessed_results = ["["
                                 + result['title']
                                 + "]("
                                 + result['url']+")\n"
                                 + result['description'] for result in results]
        # returns a list of ~10 strings formatted as [title](URL)\ndescription
        return "## Search Results\n\n" + "\n\n".join(postprocessed_results)