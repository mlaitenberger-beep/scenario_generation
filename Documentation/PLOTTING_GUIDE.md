control-Shift-V to Run this file 

# Plotting Guide: Visualizing Generated Scenarios

Complete guide for using `plot_generated.py` to analyze and interpret scenario generation results.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Command-Line Arguments](#command-line-arguments)
3. [Understanding the Output](#understanding-the-output)
4. [Common Plotting Scenarios](#common-plotting-scenarios)
5. [Customization](#customization)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Tips](#advanced-tips)

---

## Quick Start

### Basic Plot Command

```bash
python scripts/plot_generated.py \
  --samples-path results/germany_boom/samples_absolute.npy \
  --hist-csv data/germany_macro_augemented.csv \
  --history-points 24 \
  --overlap 0 \
  --output-pdf results/germany_boom/samples_plot.pdf
```

**Result**: Multi-page PDF with one page per feature, showing historical tail + generated forecasts.

**Key Parameter**: `--overlap 0` means forecast starts immediately after history. Previously the default was 6, which caused the forecast to start 6 time steps before the end of history (creating visual overlap).

---

## Command-Line Arguments

### Required Arguments

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--samples-path` | Path | Path to `.npy` file containing samples | `results/boom/samples_absolute.npy` |
| `--hist-csv` | Path | Path to historical CSV for context | `data/germany_macro.csv` |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--history-points` | Int | 24 | Number of historical points to plot before forecast |
| `--overlap` | Int | 0 | Offset for forecast x-axis (0=starts after history, 6=overlaps by 6 steps) |
| `--limit-samples` | Int | 10 | Maximum number of samples to plot (prevents clutter) |
| `--feature-names` | String | CSV headers | Comma-separated feature names for titles |
| `--output-pdf` | Path | `samples_plot.pdf` | Output PDF file path |
| `--use-absolute` | Flag | False | If set, treat samples as absolute values (skip conversion) |

---

## Understanding the Output

### PDF Structure

Each page represents **one feature** (e.g., GDP, Unemployment, Inflation):

**Example Page: GDP**

```
Title: "GDP: history vs generated"
X-axis: Time index (0 = oldest historical point)
Y-axis: Value (absolute level, e.g., billions of euros)

Plot Elements:
├─ Green dashed line: Historical data (last 24 points)
├─ Red solid lines: Generated samples (10 scenarios, 24 steps each)
│   └─ Alpha=0.4 (semi-transparent to show overlap)
└─ Legend: "history", "sample"
```
