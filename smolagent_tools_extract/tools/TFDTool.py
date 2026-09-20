class TFDTool(Tool):
    name = "time_frequency_distribution_tool"
    description = """
    Computes Time-Frequency Distributions (Cohen's Class) or Time-Scale Distributions (Affine Class).

    Supported Methods:
    - 'wvd': Wigner-Ville (Standard, high resolution)
    - 'choi_williams': Reduces interference (good for speech/bio)
    - 'rihaczek': Complex energy distribution
    - 'born_jordan': Preserves time support
    - 'zam': Cone-shaped kernel (ZAM), good for non-stationary
    - 'bertrand': Unitary Affine (Hyperbolic chirps)
    - 'unterberger': Active Wideband
    - 'affine': Scalogram analysis (Wavelet-like)

    Input: Requires the dictionary output from AudioHandlerTool.
    Output: Generates an ACTIVE Matplotlib figure. Returns a recommendation to use 'save_plot' or the raw data for advanced post-processing.
    """

    inputs = {
        "audio_dict": {
            "type": "any",
            "description": "The dictionary returned by AudioHandlerTool (must contain 'audio_samples' and 'metadata')."
        },
        "method": {
            "type": "string",
            "description": "Method: 'wvd', 'choi_williams', 'rihaczek', 'born_jordan', 'zam', 'bertrand', 'unterberger', 'affine'."
        },
        "channel": {
            "type": "integer",
            "description": "Channel index to analyze if audio is multi-channel. Default 0.",
            "nullable": True
        },
        "param": {
            "type": "number",
            "description": "Kernel parameter (Sigma for Choi-Williams, Alpha for Bertrand/ZAM). Default varies by method.",
            "nullable": True
        },
        "start_sample": {
            "type": "integer",
            "description": "The starting sample index for the analysis window. Default is 0.",
            "nullable": True
        },
        "end_sample": {
            "type": "integer",
            "description": "The ending sample index for the analysis window. If None, uses the end of the signal.",
            "nullable": True
        },
        "force_computation": {
            "type": "boolean",
            "description": "If True, overrides the O(N^2) complexity safety check for long signals (>3000 samples). Default is False.",
            "nullable": True
        },
        "return_data": {
            "type": "boolean",
            "description": "If True, returns a dictionary with raw matrix and axes instead of saving a plot. Default False.",
            "nullable": True
        }
    }
    output_type = "any"  # Can be string (message) or dict (data)

    def __init__(self):
        super().__init__()
        logging.info("TFDTool v4 (Gv9) initialized. Time/Frequency Distributions will be computed.")

    def _get_kernel(self, method, param):
        """
        Factory to generate the correct kernel function.
        Uses Robust Normalized Scaling (nu * tau/N).
        """
        p = param if param is not None else 1.0

        def choi_williams_kernel(sigma):
            def k(A, nu, tau):
                NU, TAU = np.meshgrid(nu, tau, indexing='ij')
                # Normalized product: (nu in [-0.5,0.5]) * (tau/N in [-0.5,0.5])
                # Scaling factor 100 makes sigma=1.0 meaningful for this range.
                v_tau = NU * (TAU / len(nu)) * 100.0
                return A * np.exp(- (v_tau)**2 / sigma)
            return k

        def rihaczek_kernel():
            def k(A, nu, tau):
                NU, TAU = np.meshgrid(nu, tau, indexing='ij')
                # Standard phase kernel
                return A * np.exp(1j * 2 * np.pi * NU * (TAU / 2.0))
            return k

        def born_jordan_kernel():
            def k(A, nu, tau):
                NU, TAU = np.meshgrid(nu, tau, indexing='ij')
                # Sinc(nu * tau)
                return A * np.sinc(NU * (TAU / len(nu)) * 100.0)
            return k

        def zam_kernel(a):
            def k(A, nu, tau):
                NU, TAU = np.meshgrid(nu, tau, indexing='ij')
                N = len(nu)
                # ZAM: Cone + Window
                tau_norm = TAU / N
                # Cone opens up in the ambiguity plane
                cone = np.sinc(NU * TAU * (2.0 / N) * 50.0)
                window = np.exp(-a * (tau_norm * 10)**2)
                return A * cone * window
            return k

        def bertrand_kernel(alpha):
            def k(A, nu, tau):
                NU, TAU = np.meshgrid(nu, tau, indexing='ij')
                # Hyperbolic argument
                u = 2 * np.pi * NU * (TAU / len(nu)) * alpha * 100.0
                den = np.sinh(u / 2.0)
                nonzero = np.abs(den) > 1e-9
                phi = np.ones_like(u, dtype=float)
                phi[nonzero] = (u[nonzero] / 2.0) / den[nonzero]
                return A * phi
            return k

        def unterberger_kernel(alpha):
            def k(A, nu, tau):
                NU, TAU = np.meshgrid(nu, tau, indexing='ij')
                u = 2 * np.pi * NU * (TAU / len(nu)) * alpha * 100.0
                return A * (1.0 / np.cosh(u / 2.0))
            return k

        method = method.lower()
        if method == 'wvd':
            return None
        if method == 'choi_williams':
            return choi_williams_kernel(p if param else 2.0)
        if method == 'rihaczek':
            return rihaczek_kernel()
        if method == 'born_jordan':
            return born_jordan_kernel()
        if method == 'zam':
            return zam_kernel(p if param else 2.0)
        if method == 'bertrand':
            return bertrand_kernel(p if param else 0.5)
        if method == 'unterberger':
            return unterberger_kernel(p if param else 0.5)
        return None

    def forward(self, audio_dict, method, channel=0, param=None, start_sample=0, end_sample=None, force_computation=False, return_data=False):
        try:
            # 1. Validation
            if not isinstance(audio_dict, dict) or 'audio_samples' not in audio_dict or 'metadata' not in audio_dict:
                return "Error: Input must be the dictionary output from AudioHandlerTool."

            samples = audio_dict['audio_samples']
            metadata = audio_dict['metadata']
            fs = metadata.get('sample_rate', 1.0)
            filename = metadata.get('filename', 'audio')

            # 2. Extraction
            total_len = len(samples)
            if samples.ndim > 1:
                num_channels = samples.shape[1]
                if channel >= num_channels:
                    return f"Error: Channel {channel} requested, but audio only has {num_channels} channels."
                sig_full = samples[:, channel]
                logging.info(f"TFDTool: Extracted channel {channel}/{num_channels}")
            else:
                sig_full = samples
                logging.info("TFDTool: Processing Mono signal.")

            start = start_sample if start_sample is not None else 0
            end = end_sample if end_sample is not None else total_len

            if start < 0 or end > total_len or start >= end:
                return f"Error: Invalid interval [{start}:{end}]."

            sig = sig_full[start:end]
            N = len(sig)

            # 3. Safety
            SAFETY_THRESHOLD = 4096
            if N > SAFETY_THRESHOLD:
                if not force_computation:
                    return (f"SAFETY STOP: The requested interval has {N} samples. "
                            f"Time-Frequency algorithms are O(N^2) and will generate a {N}x{N} matrix. "
                            f"This exceeds the safety threshold of {SAFETY_THRESHOLD}. "
                            f"NO COMPUTATION PERFORMED. "
                            f"Action required: Reduce the interval (end_sample - start_sample) OR set 'force_computation=True'.")
                else:
                    logging.warning(f"TFDTool: Forcing computation on large signal (N={N}). Expect high memory usage.")

            # 4. Computation
            method = method.lower()
            output_data = None
            axes_data = {}

            if method == 'affine':
                engine = _AffineEngine(fs=fs)
                scales = np.logspace(0.1, 1.2, 50)
                output_data = engine.compute(sig, scales)
                axes_data = {
                    'y_axis': scales,
                    'y_label': 'Scale',
                    'x_axis': np.arange(N)/fs,
                    'type': 'time-scale'
                }
            else:
                kernel = self._get_kernel(method, param)
                engine = _CohenEngine(kernel_func=kernel)
                output_data, freqs = engine.compute(sig, fs=fs)
                axes_data = {
                    'y_axis': freqs,
                    'y_label': 'Frequency (Hz)',
                    'x_axis': np.arange(N)/fs,
                    'type': 'time-frequency'
                }

            # 5. Return Logic
            if return_data:
                # Return raw objects for advanced downstream plotting
                return {
                    "tfd_matrix": output_data,  # Complex matrix (Time x Freq) or (Scale x Time)
                    "axes": axes_data,
                    "metadata": {
                        "fs": fs,
                        "n_samples": N,
                        "method": method,
                        "param": param,
                        "source_file": filename
                    }
                }

            # 6. Default Visualization Logic (Linear Freq, Log Magnitude)
            mag_data = np.abs(output_data)

            extent = [0, N/fs, axes_data['y_axis'][0], axes_data['y_axis'][-1]]

            if method != 'affine':
                # Slicing Logic for Cohen's Class (Wigner-Ville)
                # The FFT of the autocorrelation has N bins.
                # Bins 0 to N/2-1 correspond to positive correlation frequencies 0 to Fs/2.
                # Bins N/2 to N-1 correspond to negative correlation frequencies.
                # For an analytic signal, we expect energy mostly in the positive range.
                # We slice the first half (Positive Freqs).

                # Note: fftshift was applied in engine, so 0 is in the middle.
                # Middle to End = Positive Frequencies.
                mid_point = mag_data.shape[1] // 2  # Axis 1 is Freq

                # Slice positive frequencies
                mag_data = mag_data[:, mid_point:]
                extent[2] = 0  # Freq start 0
                extent[3] = fs/2  # Freq end Nyquist

            # dB Scale
            display_data = 20 * np.log10(mag_data + 1e-10)
            max_val = np.max(display_data)
            display_data = np.maximum(display_data, max_val - 80)

            # Transpose for plotting (Time on X, Freq on Y)
            # Matrix is (Time, Freq), imshow needs (Freq, Time)
            display_data = display_data.T

            plt.figure(figsize=(10, 6))
            plt.imshow(display_data, aspect='auto', origin='lower',
                       extent=extent, cmap='jet')

            title_str = f"{method.upper()} (dB) | {filename} | Ch {channel}"
            plt.title(title_str)
            plt.ylabel(axes_data['y_label'])
            plt.xlabel("Time (s)")
            plt.colorbar(label='Magnitude (dBFS)')
            plt.tight_layout()

            rec_filename = f"tfd_{filename.replace('.','_')}_{method}_{start}_{end}.png"
            return (f"TFD computed. Plotted on active figure. "
                    f"Use 'save_plot' to save as '{rec_filename}'.")

        except Exception as e:
            return f"Error computing TFD: {str(e)}"