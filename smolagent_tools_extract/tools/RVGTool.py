class RVGTool(Tool):
    name = "random_variate_generator"
    description = (
        "Generates lists of pseudo-random numbers from specified statistical distributions. "
        "It uses a high-precision Linear Congruential Generator (LCG) as its foundation. "
        "Available distributions: 'uniform', 'normal', 'exponential'."
    )
    inputs = {
        "distribution": {
            "type": "string",
            "description": "The name of the probability distribution to sample from. Supported: 'uniform', 'normal', 'exponential'."
        },
        "size": {
            "type": "integer",
            "description": "The number of random numbers to generate."
        },
        "seed": {
            "type": "integer",
            "description": "An integer used to initialize the random number generator. Using the same seed will produce the exact same sequence of numbers again, which is useful for reproducibility.",
            "nullable": True
        },
        "distribution_params": {
            "type": "object",
            "description": "A dictionary of parameters specific to the chosen distribution. "
                           "For 'normal', use {'mu': <mean>, 'sigma': <standard_deviation>}. "
                           "For 'exponential', use {'lambda_': <rate>}. "
                           "For 'uniform', this can be omitted.",
            "nullable": True
        }
    }
    output_type = "object"

    def __init__(self):
        super().__init__()
        # Default LCG parameters. These are well-known and provide good results.
        self.m = 2**100
        self.a = 1664525
        self.c = 1013904223
        self.default_seed = 42
        self.precision = 200
        # The tool holds its own internal generator instances
        self.lcg = LCG(m=self.m, a=self.a, c=self.c, seed=self.default_seed, precision=self.precision)
        self.rv_gen = RandomVariateGenerator(generator=self.lcg)
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("RVGTool initialized with default high-precision LCG.")

    def forward(self, distribution: str, size: int, seed: int = None, distribution_params: dict = None):
        if distribution_params is None:
            distribution_params = {}

        try:
            # If a new seed is provided, re-initialize the generator for this specific call
            if seed is not None:
                logging.info(f"Re-initializing RVGTool with provided seed: {seed}")
                temp_lcg = LCG(m=self.m, a=self.a, c=self.c, seed=seed, precision=self.precision)
                generator_to_use = RandomVariateGenerator(generator=temp_lcg)
            else:
                # Otherwise, use the tool's persistent internal generator
                generator_to_use = self.rv_gen

            # Generate the random numbers
            numbers = generator_to_use.rvs(
                distribution=distribution,
                size=size,
                **distribution_params
            )

            return {
                "status": "success",
                "generated_numbers": numbers,
                "message": f"Successfully generated {size} random variates from a '{distribution}' distribution."
            }

        except Exception as e:
            logging.error(f"An error occurred in RVGTool: {e}")
            return {
                "status": "error",
                "message": str(e)
            }