class LogFreqPlotTool(Tool):
    name = "log_freq_plot_tool"
    description = """
    Generates a Log-Frequency (Octave-scale) Spectrogram from raw TFD data.
    Useful for visualizing musical tones, speech, or wideband signals.
    Resamples the linear Time-Frequency matrix onto a logarithmic frequency grid.

    Returns a recommendation to use 'save_plot'.
    """
    inputs = {
        "tfd_data": {
            "type": "any",
            "description": "The dictionary containing 'tfd_matrix' and 'axes' (from TFDTool or TFDDataTool)."
        },
        "f_min": {
            "type": "number",
            "description": "Minimum frequency for the log axis (Hz). Default 20.",
            "nullable": True
        },
        "f_max": {
            "type": "number",
            "description": "Maximum frequency (Hz). Default is Nyquist (Fs/2).",
            "nullable": True
        },
        "num_bins": {
            "type": "integer",
            "description": "Height of the resulting image in pixels/bins. Default 500.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        logging.info("LogFreqPlotTool v1 (Gv1) initialized. Time/Frequency Distributions Log Plot will be generated.")

    def forward(self, tfd_data, f_min=20.0, f_max=None, num_bins=500):
        try:
            # 1. Unpack Data
            if not isinstance(tfd_data, dict) or 'tfd_matrix' not in tfd_data or 'axes' not in tfd_data:
                return "Error: Input must be a valid TFD data dictionary."

            tfd_matrix = tfd_data['tfd_matrix']  # Shape: (Time, LinearFreq)
            axes = tfd_data['axes']
            meta = tfd_data.get('metadata', {})

            lin_freqs = axes['y_axis']  # The original linear frequency axis
            time_axis = axes['x_axis']

            if axes.get('type') != 'time-frequency':
                return "Error: Log-Frequency plotting is only valid for Time-Frequency data (Cohen Class), not Affine."

            # 2. Define Log Grid
            # Ensure f_min > 0 to avoid log(0)
            f_min = max(f_min, 1.0)

            # Default f_max to Nyquist if not provided
            max_available_freq = lin_freqs[-1]
            if f_max is None or f_max > max_available_freq:
                f_max = max_available_freq

            # Create the target Logarithmic Frequency Axis
            # np.logspace uses base 10. Start and stop are powers of 10.
            log_freqs = np.logspace(np.log10(f_min), np.log10(f_max), num_bins)

            # 3. Interpolation (Resampling)
            # We need to map the columns of tfd_matrix (Linear Freq) to log_freqs.
            # interp1d creates a function f(freq) -> amplitude
            # axis=1 means we interpolate along the frequency dimension for all time steps at once.

            # Input frequencies must be strictly increasing.
            # TFDTool returns [-Fs/2 ... 0 ... Fs/2]. We need to slice only Positive.

            # Find index of 0 Hz
            zero_idx = len(lin_freqs) // 2
            pos_lin_freqs = lin_freqs[zero_idx:]
            pos_tfd_matrix = tfd_matrix[:, zero_idx:]  # Slice positive half

            # Handle potential tiny mismatch in length due to odd/even N
            if pos_tfd_matrix.shape[1] != len(pos_lin_freqs):
                min_len = min(pos_tfd_matrix.shape[1], len(pos_lin_freqs))
                pos_tfd_matrix = pos_tfd_matrix[:, :min_len]
                pos_lin_freqs = pos_lin_freqs[:min_len]

            # Create Interpolator
            # bounds_error=False, fill_value=0 handles frequencies requested outside the computed range
            interpolator = interp1d(pos_lin_freqs, np.abs(pos_tfd_matrix), kind='linear',
                                    axis=1, bounds_error=False, fill_value=0.0)

            # Generate the new Log-Frequency Matrix
            # Shape: (Time, NewLogFreqs)
            log_tfd_matrix = interpolator(log_freqs)

            # 4. Visualization (dB Scale)
            display_data = 20 * np.log10(log_tfd_matrix + 1e-10)
            max_val = np.max(display_data)
            display_data = np.maximum(display_data, max_val - 80)

            # Transpose for Plotting: (Time, Freq) -> (Freq, Time)
            display_data = display_data.T

            plt.figure(figsize=(10, 6))

            # We use pcolormesh or imshow.
            # Since we resampled to a grid that is linear-in-log-space, we can use imshow
            # and just label the ticks manually, OR use extent with log scale.
            # Using extent with imshow on log axes is tricky in Matplotlib.
            # Easiest robust way: imshow with custom Y-ticks.

            plt.imshow(display_data, aspect='auto', origin='lower', cmap='jet')

            # Handle Ticks
            # Y-axis is now indices 0 to num_bins-1. We map them back to Hz for labels.
            num_ticks = 10
            tick_indices = np.linspace(0, num_bins-1, num_ticks)
            tick_labels = [f"{val:.0f}" for val in np.logspace(np.log10(f_min), np.log10(f_max), num_ticks)]

            plt.yticks(tick_indices, tick_labels)

            # X-axis (Time)
            # Extent is not strictly valid for Y here, so we set X ticks manually too or rely on aspect='auto'
            # Let's try to set X extent correctly
            plt.xlim(0, display_data.shape[1])
            # Map X pixels to Time
            t_min, t_max = time_axis[0], time_axis[-1]
            x_ticks = np.linspace(0, display_data.shape[1], 8)
            x_labels = [f"{t:.2f}" for t in np.linspace(t_min, t_max, 8)]
            plt.xticks(x_ticks, x_labels)

            method_name = meta.get('method', 'TFD').upper()
            fname = meta.get('source_file', 'audio')

            plt.title(f"{method_name} (Log-Freq) | {fname}")
            plt.ylabel("Frequency (Hz) [Log Scale]")
            plt.xlabel("Time (s)")
            plt.colorbar(label='Magnitude (dBFS)')
            plt.tight_layout()

            rec_filename = f"log_tfd_{fname}_{method_name}.png"
            return (f"Log-Frequency TFD plotted on active figure. "
                    f"Use 'save_plot' to save as '{rec_filename}'.")

        except Exception as e:
            return f"Error creating Log-Freq plot: {str(e)}"