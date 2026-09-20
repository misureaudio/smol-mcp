class SavePlotTool(Tool):
    name = "save_plot"
    description = "Saves the currently active Matplotlib plot to a file in the secure 'plots' directory."
    inputs = {"output_filename": {"type": "string", "description": "The desired filename for the output PNG."}}
    output_type = "string"

    def __init__(self, safe_dir: str):
        super().__init__()
        self.safe_dir = safe_dir
        if not os.path.exists(self.safe_dir):
            os.makedirs(self.safe_dir, exist_ok=True)
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info(f"SavePlotTool initialized.\n*** ✅ ---> Plots will be saved in '{os.path.abspath(self.safe_dir)}'")

    def forward(self, output_filename: str) -> str:
        if not output_filename.lower().endswith('.png'):
            return "Error: The output_filename must end with .png."
        full_path = os.path.join(self.safe_dir, os.path.basename(output_filename))
        safe_path = full_path.replace('\\', '/')
        try:
            plt.savefig(safe_path)
            plt.close('all')
            return f"Successfully saved plot to {full_path}"
        except Exception as e:
            return f"Error while saving plot: {e}"