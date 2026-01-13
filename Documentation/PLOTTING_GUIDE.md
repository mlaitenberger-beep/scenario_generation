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

```powershell
python scripts/plot_generated.py `
  --samples-path results/germany_boom/samples_absolute.npy `
  --hist-csv data/germany_macro_augemented.csv `
  --output-pdf results/germany_boom/samples_plot.pdf
```

**Result**: Multi-page PDF with one page per feature, showing historical tail + generated forecasts.

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
| `--overlap` | Int | 6 | Number of overlapping points between history and forecast |
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

### Interpreting the Plot

#### 1. **Continuity Check**

**Good**: Generated samples start near the last historical value and transition smoothly.

**Bad**: Large jump/discontinuity at the history-forecast boundary.

**Why it matters**: Indicates whether the model respects initial conditions.

**Example**:
```
History ends at: 1050
First forecast step: 1048 (good), 900 (bad - discontinuity)
```

---

#### 2. **Range Check**

**Good**: Samples span a reasonable range around historical trends.

**Bad**: All samples collapse to a single line (no diversity) or explode to unrealistic values.

**Why it matters**: Model should capture uncertainty, not just deterministic predictions.

**Example**:
```
Historical range: [1000, 1100]
Generated range: [1020, 1150] (good - reasonable expansion)
Generated range: [10, 5000] (bad - unrealistic)
```

---

#### 3. **Trend Consistency**

**Good**: If history shows upward trend, samples follow similar patterns (unless stressed otherwise).

**Bad**: Samples exhibit completely different dynamics (e.g., oscillations when history is smooth).

**Why it matters**: Model should learn underlying data patterns.

---

#### 4. **Stress Conditioning Validation**

If you used `--stressed-features`, check if:
- Stressed features follow the specified stress values
- Non-stressed features vary naturally around the stressed trajectory

**Example**: Stress GDP to grow 5% but leave Unemployment free.
- GDP samples should cluster around 5% growth path
- Unemployment should show realistic variation (not forced)

---

### Visual Quality Indicators

| Indicator | Good | Bad |
|-----------|------|-----|
| **Smoothness** | Gradual changes, no spikes | Erratic jumps, noise |
| **Diversity** | Samples spread out | All samples identical |
| **Boundary** | Smooth transition at history-forecast join | Discontinuity, gap |
| **Realism** | Values within plausible range | Negative GDP, 200% unemployment |

---

## Common Plotting Scenarios

### Scenario 1: Compare Baseline vs. Stress

**Baseline plot**:
```powershell
python scripts/plot_generated.py `
  --samples-path results/baseline/samples_absolute.npy `
  --hist-csv data/germany_macro.csv `
  --output-pdf results/baseline_plot.pdf
```

**Stress plot**:
```powershell
python scripts/plot_generated.py `
  --samples-path results/stress_boom/samples_absolute.npy `
  --hist-csv data/germany_macro.csv `
  --output-pdf results/stress_plot.pdf
```

**Analysis**: Open both PDFs side-by-side. Do stressed scenarios diverge from baseline as expected?

---

### Scenario 2: Plot Many Samples (50+)

**Warning**: Plotting 200 samples creates visual clutter.

**Solution**: Limit to representative subset:
```powershell
--limit-samples 20
```

**Alternative**: Plot percentiles instead (requires custom script—see Advanced Tips).

---

### Scenario 3: Custom Feature Names

If CSV headers are generic (e.g., `V1`, `V2`, `V3`), override with meaningful names:

```powershell
--feature-names "GDP,Unemployment,Inflation,InterestRate,ExchangeRate,..."
```

**Result**: Plot titles show "GDP: history vs generated" instead of "V1: history vs generated".

---

### Scenario 4: Longer Historical Context

Default shows last 24 points. To show more:

```powershell
--history-points 48
```

**Use case**: Visualize longer-term trends before forecast.

---

### Scenario 5: Adjust Overlap

**Overlap** controls how many steps history and forecast share (for smooth visual connection).

**More overlap** (better continuity):
```powershell
--overlap 12
```

**Less overlap** (clearer separation):
```powershell
--overlap 2
```

**Default**: 6 (balanced)

---

## Customization

### Changing Colors

Edit `plot_generated.py` line ~60:

```python
# History: green dashed
plt.plot(x_history, history[:, feat_idx], linestyle="dashed", color="green", label="history")

# Samples: red solid
plt.plot(x_forecast, samples[i, :, feat_idx], linestyle="solid", color="red", alpha=0.4)
```

**Example customizations**:
- Change history to blue: `color="blue"`
- Change samples to orange: `color="orange"`
- Increase transparency: `alpha=0.2`
- Use different line styles: `linestyle="dotted"`, `linestyle="dashdot"`

---

### Adding Grid Lines

Add after line ~70:

```python
plt.grid(True, alpha=0.3, linestyle='--')
```

---

### Changing Figure Size

Modify line ~55:

```python
plt.figure(figsize=(12, 3))  # default
plt.figure(figsize=(16, 5))  # wider, taller
```

---

### Saving Individual PNGs

Replace `pdf.savefig()` with:

```python
png_path = os.path.join(os.path.dirname(output_pdf), f"feature_{feat_idx}.png")
plt.savefig(png_path, dpi=300)
```

---

## Troubleshooting

### Issue: Samples appear as straight lines

**Cause**: Plotting relative changes instead of absolute values.

**Solution**: Use `--use-absolute` flag or pass `samples_absolute.npy`:
```powershell
--samples-path results/boom/samples_absolute.npy --use-absolute
```

---

### Issue: History and forecast don't connect smoothly

**Cause**: Incorrect overlap calculation or data discontinuity.

**Fix**: Ensure `--overlap` matches the model's overlap configuration (typically 6).

---

### Issue: All samples look identical

**Cause**: Model collapsed (not enough diversity), or sampling steps too low.

**Solutions**:
1. Retrain model with higher capacity (`d_model`, layers)
2. Increase `--sampling-steps` during generation (e.g., 500)
3. Check if model checkpoint is too early (try later epoch)

---

### Issue: Values are scaled incorrectly

**Cause**: CSV has scaling issues (e.g., GDP in thousands vs. billions).

**Solution**: Verify CSV units are consistent. If needed, rescale externally:
```python
df = pd.read_csv("data/hist.csv")
df["GDP"] = df["GDP"] / 1000  # billions → trillions
df.to_csv("data/hist_scaled.csv", index=False)
```

---

### Issue: PDF has too many pages

**Cause**: `feature_size=11` → 11 pages.

**Solution**: Plot subset of features by filtering before plotting:
```python
# In plot script, add:
features_to_plot = [0, 1, 2]  # GDP, Unemployment, Inflation only
for feat_idx in features_to_plot:
    # ... plotting code
```

---

## Advanced Tips

### Tip 1: Plot Confidence Bands

Calculate percentiles across samples and plot shaded regions:

```python
import numpy as np

lower = np.percentile(samples, 5, axis=0)   # 5th percentile
upper = np.percentile(samples, 95, axis=0)  # 95th percentile
median = np.median(samples, axis=0)

plt.fill_between(x_forecast, lower[:, feat_idx], upper[:, feat_idx], 
                 alpha=0.3, color='red', label='90% confidence')
plt.plot(x_forecast, median[:, feat_idx], color='red', linewidth=2, label='median')
```

**Result**: Shaded band showing forecast uncertainty.

---

### Tip 2: Add Reference Lines

Mark key thresholds (e.g., 0% growth, recession threshold):

```python
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
plt.text(x_forecast[0], 0, "Zero growth", fontsize=8, va='bottom')
```

---

### Tip 3: Compare Stressed vs. Unstressed Features

**Custom script**: Load both `baseline` and `stressed` samples, plot on same axes:

```python
baseline = np.load("results/baseline/samples_absolute.npy")
stressed = np.load("results/stressed/samples_absolute.npy")

plt.figure()
for i in range(10):
    plt.plot(baseline[i, :, 0], color='blue', alpha=0.3, label='baseline' if i == 0 else None)
    plt.plot(stressed[i, :, 0], color='red', alpha=0.3, label='stressed' if i == 0 else None)
plt.legend()
plt.title("GDP: Baseline vs. Stressed")
plt.savefig("comparison.png")
```

---

### Tip 4: Interactive Plots

Use Plotly for zoom/pan/hover:

```python
import plotly.graph_objects as go

fig = go.Figure()
for i in range(num_samples):
    fig.add_trace(go.Scatter(x=x_forecast, y=samples[i, :, 0], 
                             mode='lines', name=f'Sample {i}'))
fig.update_layout(title="GDP Scenarios", xaxis_title="Time", yaxis_title="Value")
fig.write_html("interactive_plot.html")
```

**Open in browser**: Zoom, pan, and inspect individual sample values.

---

### Tip 5: Animated Scenario Evolution

Create GIF showing scenarios evolving over time:

```python
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
line, = ax.plot([], [], color='red')

def init():
    ax.set_xlim(0, seq_len)
    ax.set_ylim(samples.min(), samples.max())
    return line,

def update(frame):
    line.set_data(range(frame+1), samples[0, :frame+1, 0])
    return line,

anim = FuncAnimation(fig, update, frames=seq_len, init_func=init, blit=True)
anim.save("scenario_evolution.gif", writer='pillow', fps=5)
```

---

## Example Workflows

### Workflow 1: Validate Stress Conditioning

1. Generate stressed scenarios
2. Plot with custom feature names:
   ```powershell
   --feature-names "GDP,Unemployment,Inflation"
   ```
3. Open PDF and inspect features 0, 1, 2 (GDP, Unemployment, Inflation)
4. Verify: Do red lines follow the stress trajectory in the latter half of the forecast?

---

### Workflow 2: Model Selection via Visual Inspection

Generate plots for multiple checkpoints:

```powershell
# Checkpoint 2
python scripts/plot_generated.py --samples-path results/ckpt2/samples_absolute.npy --output-pdf ckpt2_plot.pdf ...

# Checkpoint 5
python scripts/plot_generated.py --samples-path results/ckpt5/samples_absolute.npy --output-pdf ckpt5_plot.pdf ...

# Checkpoint 10
python scripts/plot_generated.py --samples-path results/ckpt10/samples_absolute.npy --output-pdf ckpt10_plot.pdf ...
```

**Compare**: Which checkpoint produces the most realistic/diverse scenarios?

---

### Workflow 3: Presentation-Ready Plots

1. Limit to 5 representative samples:
   ```powershell
   --limit-samples 5
   ```
2. Increase DPI for high-res output (edit script):
   ```python
   pdf.savefig(dpi=300)
   ```
3. Add custom title/annotations (edit script):
   ```python
   plt.suptitle("Germany Macro Stress Scenarios - Boom Case", fontsize=14)
   ```

---

## Exporting for Other Tools

### Export to Excel

```python
import pandas as pd
import numpy as np

samples = np.load("results/boom/samples_absolute.npy")
df = pd.DataFrame(samples[0])  # First sample
df.columns = ["GDP", "Unemployment", "Inflation", ...]
df.to_excel("scenario_0.xlsx", index=False)
```

---

### Export to CSV for Tableau/Power BI

```python
# Reshape to long format
data = []
for i in range(num_samples):
    for t in range(seq_len):
        for f in range(feature_size):
            data.append({
                "sample_id": i,
                "time_step": t,
                "feature": feature_names[f],
                "value": samples[i, t, f]
            })
df = pd.DataFrame(data)
df.to_csv("scenarios_long.csv", index=False)
```

**Import into Tableau**: Use "sample_id" as dimension, "value" as measure, "feature" as color.

---

## Best Practices

1. **Always use `samples_absolute.npy`** for plotting (unless specifically analyzing relative changes)
2. **Limit samples to 5–20** for clarity in static plots
3. **Use percentile bands** for summarizing many samples
4. **Label axes and features clearly** (override CSV headers if needed)
5. **Save high-res PDFs** (300 DPI) for reports/presentations
6. **Validate stress conditioning visually** before downstream use

---

## FAQ

### Q: Can I plot only certain features?

**A**: Modify the script to loop over a subset:
```python
features_to_plot = [0, 2, 5]  # GDP, Inflation, ExchangeRate
for feat_idx in features_to_plot:
    # ... existing plotting code
```

---

### Q: How do I overlay multiple runs on one plot?

**A**: Custom script required. Load multiple `.npy` files and plot with different colors:
```python
run1 = np.load("results/run1/samples_absolute.npy")
run2 = np.load("results/run2/samples_absolute.npy")

plt.figure()
for i in range(10):
    plt.plot(run1[i, :, 0], color='blue', alpha=0.3)
    plt.plot(run2[i, :, 0], color='red', alpha=0.3)
plt.legend(["Run 1", "Run 2"])
plt.savefig("comparison.png")
```

---

### Q: What if CSV headers are missing?

**A**: Use `--feature-names` to provide them manually:
```powershell
--feature-names "Var1,Var2,Var3,..."
```

---

## Next Steps

- **Quantitative validation**: Compute MAE, RMSE, correlation against holdout data
- **Distributional metrics**: Compare generated vs. historical distributions (KL divergence, Wasserstein)
- **Integrate with downstream models**: Feed scenarios into risk models, portfolio optimizers, etc.

---

**Questions?**
- See USER_GUIDE.md for end-to-end workflows
- See ARCHITECTURE.md for technical details

---

**Last Updated**: January 13, 2026
