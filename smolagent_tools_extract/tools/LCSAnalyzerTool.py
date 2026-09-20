class LCSAnalyzerTool(Tool):
    """
    A tool to analyze two texts and find their Longest Common Subsequence (LCS).
    It uses advanced text normalization (stemming or lemmatization) and a
    space-efficient algorithm (Hirschberg's algorithm) for the calculation.
    """
    name = "lcs_analyzer"
    description = (
        "Calculates the Longest Common Subsequence (LCS) of two input texts. "
        "It first normalizes the texts by converting them into a sequence of root words "
        "using either lemmatization or stemming. Then, it uses an efficient algorithm "
        "to find the longest sequence of words common to both texts."
    )
    inputs = {
        "text1": {"type": "string", "description": "The first text to be analyzed."},
        "text2": {"type": "string", "description": "The second text to be analyzed."},
        "use_lemmatization": {"type": "boolean", "description": "If True (default), uses lemmatization for text normalization. If False, uses stemming.", "nullable": True},
        "parallel": {"type": "boolean", "description": "If True (default), processes the two texts in parallel to speed up normalization.", "nullable": True},
        "return_sequence": {"type": "boolean", "description": "If True (default), returns the actual LCS sequence of words. If False, returns only its length.", "nullable": True}
    }
    output_type = "string"  # The output will be a JSON string.

    # Class-level variables to hold the loaded models, so they are loaded only once.
    _nlp_spacy = None
    _nltk_stopwords = None

    def __init__(self):
        super().__init__()
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        # Use the correct class name to check if models are loaded.
        if LCSAnalyzerTool._nlp_spacy is None:
            self._setup_dependencies()

    def _setup_dependencies(self):
        """Downloads and loads all required NLTK and spaCy models."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logging.info("Downloading NLTK data (punkt, stopwords)...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)

        try:
            # Assign to the class variable directly.
            LCSAnalyzerTool._nlp_spacy = spacy.load("en_core_web_sm")
            logging.info("spaCy model 'en_core_web_sm' loaded.")
        except OSError:
            logging.info("spaCy model 'en_core_web_sm' not found. Downloading...")
            spacy.cli.download("en_core_web_sm")
            LCSAnalyzerTool._nlp_spacy = spacy.load("en_core_web_sm")

        LCSAnalyzerTool._nltk_stopwords = set(stopwords.words('english'))

    def _hirschberg_lcs(self, X, Y):
        """Hirschberg's algorithm: O(mn) time, O(min(m,n)) space."""
        def lcs_length_forward(X, Y):
            m, n = len(X), len(Y)
            curr = [0] * (n + 1)
            for i in range(1, m + 1):
                prev = curr[:]
                for j in range(1, n + 1):
                    if X[i-1] == Y[j-1]:
                        curr[j] = prev[j-1] + 1
                    else:
                        curr[j] = max(curr[j-1], prev[j])
            return curr

        def lcs_length_backward(X, Y):
            return lcs_length_forward(X[::-1], Y[::-1])[::-1]

        if not X or not Y:
            return 0, []
        if len(X) == 1:
            return (1, [X[0]]) if X[0] in Y else (0, [])

        mid = len(X) // 2
        left_lengths = lcs_length_forward(X[:mid], Y)
        right_lengths = lcs_length_backward(X[mid:], Y)
        partition = max(range(len(Y) + 1), key=lambda k: left_lengths[k] + right_lengths[k])
        left_len, left_seq = self._hirschberg_lcs(X[:mid], Y[:partition])
        right_len, right_seq = self._hirschberg_lcs(X[mid:], Y[partition:])
        return left_len + right_len, left_seq + right_seq

    def _calculate_lcs_length_optimized(self, seq1, seq2):
        """Space-optimized LCS length calculation: O(min(m,n)) space."""
        if len(seq1) > len(seq2):
            seq1, seq2 = seq2, seq1
        m, n = len(seq1), len(seq2)
        prev = [0] * (m + 1)
        for j in range(1, n + 1):
            curr = [0] * (m + 1)
            for i in range(1, m + 1):
                if seq1[i-1] == seq2[j-1]:
                    curr[i] = prev[i-1] + 1
                else:
                    curr[i] = max(curr[i-1], prev[i])
            prev = curr
        return prev[m]

    def _process_texts_parallel(self, texts, use_lemmatization):
        """Process multiple texts in parallel using multiprocessing."""
        if len(texts) < 2 or sum(len(t) for t in texts) < 10000:
            processor = _TextProcessor(LCSAnalyzerTool._nlp_spacy, LCSAnalyzerTool._nltk_stopwords, use_lemmatization)
            return [processor.process(text) for text in texts]

        try:
            # Use functools.partial to create a function with the 'use_lemmatization' argument fixed.
            worker_func = partial(_lcs_worker_process, use_lemmatization=use_lemmatization)
            with Pool(processes=min(len(texts), os.cpu_count() or 2)) as pool:
                results = pool.map(worker_func, texts)
            return results
        except Exception as e:
            logging.warning(f"Multiprocessing pool failed with error: {e}. Falling back to serial processing.")
            processor = _TextProcessor(LCSAnalyzerTool._nlp_spacy, LCSAnalyzerTool._nltk_stopwords, use_lemmatization)
            return [processor.process(text) for text in texts]

    def forward(self, text1: str, text2: str, use_lemmatization: bool = True, parallel: bool = True, return_sequence: bool = True) -> str:
        """Executes the LCS analysis."""
        # Process texts
        if parallel:
            words1, words2 = self._process_texts_parallel([text1, text2], use_lemmatization)
        else:
            processor = _TextProcessor(LCSAnalyzerTool._nlp_spacy, LCSAnalyzerTool._nltk_stopwords, use_lemmatization)
            words1 = processor.process(text1)
            words2 = processor.process(text2)

        # Calculate LCS
        if return_sequence:
            lcs_len, lsc_seq = self._hirschberg_lcs(words1, words2)
        else:
            lcs_len = self._calculate_lcs_length_optimized(words1, words2)
            lsc_seq = None

        results = {
            'normalization_method': "Lemmatization" if use_lemmatization else "Stemming",
            'text1_normalized_word_count': len(words1),
            'text2_normalized_word_count': len(words2),
            'lcs_length': lcs_len,
            'lcs_sequence': lsc_seq
        }

        return json.dumps(results, indent=2)