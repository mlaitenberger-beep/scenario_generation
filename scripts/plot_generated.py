"""Plot generated samples against recent historical data.

This script loads the generated samples (numpy array) and a historical CSV, then
creates per-feature line plots comparing the historical tail with each
generated forecast. Outputs a single PDF with one page per feature.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def parse_feature_names(arg_value, df_columns, feature_size):
    """Return list of feature names using CLI override or CSV headers."""
    if arg_value:
        names = [name.strip() for name in arg_value.split(',') if name.strip()]
        return names
    if df_columns is not None and len(df_columns) >= feature_size:
        return list(df_columns[:feature_size])
    return [f"feature_{i}" for i in range(feature_size)]


def main():
    parser = argparse.ArgumentParser(description="Plot generated samples versus historical tail")
    parser.add_argument("--samples-path", required=True, help="Path to samples.npy or samples_absolute.npy produced by run_handler_real.py")
    parser.add_argument("--hist-csv", required=True, help="Path to historical data CSV")
    parser.add_argument("--history-points", type=int, default=24, help="Number of historical points to show before forecasts")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap between history and generated forecast")
    parser.add_argument("--limit-samples", type=int, default=10, help="Max number of generated samples to plot")
    parser.add_argument("--feature-names", default=None, help="Comma-separated feature names; defaults to CSV headers or feature_{i}")
    parser.add_argument("--output-pdf", default=None, help="Output PDF path; defaults beside samples.npy as samples_plot.pdf")
    parser.add_argument("--use-absolute", action="store_true", help="If set, treat samples as absolute values (default: relative changes)")
    args = parser.parse_args()

    samples_path = os.path.abspath(args.samples_path)
    hist_csv = os.path.abspath(args.hist_csv)
    output_pdf = args.output_pdf or os.path.join(os.path.dirname(samples_path), "samples_plot.pdf")

    samples = np.load(samples_path)
    if samples.ndim != 3:
        raise ValueError(f"Expected samples of shape (n, seq_len, feat); got {samples.shape}")

    # Keep only requested number of samples
    samples = samples[: args.limit_samples]
    num_samples, seq_len, feature_size = samples.shape

    df_hist = pd.read_csv(hist_csv, header=0, dtype=float)
    # Use most recent history_points rows and align feature count
    history = df_hist.iloc[-args.history_points :, :feature_size].to_numpy()

    # If samples are relative changes, convert to absolute values
    if not args.use_absolute:
        last_hist_value = df_hist.iloc[-1, :feature_size].to_numpy()
        samples_abs = np.zeros_like(samples)
        for i in range(num_samples):
            current = last_hist_value.copy()
            for t in range(seq_len):
                current = current * (1.0 + samples[i, t, :])
                samples_abs[i, t, :] = current
        samples = samples_abs

    feature_names = parse_feature_names(args.feature_names, df_hist.columns if hasattr(df_hist, "columns") else None, feature_size)

    x_history = np.arange(history.shape[0])
    x_forecast = np.arange(history.shape[0] - args.overlap, history.shape[0] - args.overlap + seq_len)

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        for feat_idx in range(feature_size):
            plt.figure(figsize=(12, 3))
            plt.plot(x_history, history[:, feat_idx], linestyle="dashed", color="green", label="history")
            for i in range(num_samples):
                plt.plot(
                    x_forecast,
                    samples[i, :, feat_idx],
                    linestyle="solid",
                    color="red",
                    alpha=0.4,
                    label="sample" if i == 0 else None,
                )
            plt.title(f"{feature_names[feat_idx]}: history vs generated")
            plt.xlabel("time index")
            plt.ylabel("value")
            plt.legend(loc="upper left")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"Saved plots to {output_pdf}")


if __name__ == "__main__":
    sys.exit(main())
