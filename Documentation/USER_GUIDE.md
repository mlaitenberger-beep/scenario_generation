control-Shift-V to Run this file 
# Scenario Generation User Guide

Complete step-by-step instructions for generating and visualizing macroeconomic stress scenarios.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Running Your First Scenario](#running-your-first-scenario)
5. [Advanced Usage](#advanced-usage)
6. [Interpreting Results](#interpreting-results)
7. [Common Workflows](#common-workflows)
8. [FAQ](#faq)

---

## Prerequisites

### System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux
- **Python**: Version 3.8 or higher
- **Memory**: 8GB RAM minimum (16GB+ recommended for GPU)
- **Storage**: 5GB free space (for model checkpoints and results)
- **GPU** (Optional but recommended): NVIDIA GPU with CUDA support for faster training/inference

### Required Software

1. **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/)
2. **Git** (optional): For cloning repositories
3. **Text Editor**: VS Code recommended for editing configs

---

## Installation

### Step 1: Navigate to Project Folder

```powershell
cd "c:\Users\mlaitenberger\OneDrive - Deloitte (O365D)\Desktop\Scenario_Project\scenario_generation"
```

### Step 2: Install Python Dependencies

```powershell
pip install torch numpy pandas scikit-learn matplotlib pyyaml
```

**For GPU support** (NVIDIA only):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Verify Installation

```powershell
python scripts/check_imports_deps.py
```

Expected output:
```
✓ torch imported successfully
✓ numpy imported successfully
✓ pandas imported successfully
✓ sklearn imported successfully
✓ matplotlib imported successfully
✓ yaml imported successfully
All dependencies installed correctly!
```

---

## Data Preparation

### Historical Data Format

Your historical CSV must have:
- **First row**: Feature names (header)
- **Subsequent rows**: Time series values (one row per time step)
- **Columns**: One per feature (e.g., GDP, Unemployment, Inflation)

**Example** (`data/germany_macro_augemented.csv`):
```csv
GDP,Unemployment,Inflation,InterestRate,ExchangeRate,...
1000.5,5.2,2.1,1.5,1.18,...
1015.3,5.1,2.3,1.6,1.19,...
1020.8,5.0,2.4,1.7,1.20,...
...
```

**Requirements**:
- All values must be numeric (float or int)
- No missing values (NaN)
- At least `seq_length + 50` rows recommended for training

### Stress Scenario Format

Your stress CSV defines the conditional values you want to impose:
- **Same format** as historical CSV
- **Length**: Typically equals `seq_length` or the stressed portion
- **Values**: Absolute values (not relative changes)

**Example** (`data/germany_stress_boom.csv`):
```csv
GDP,Unemployment,Inflation,InterestRate,ExchangeRate,...
1050.0,4.5,2.5,1.8,1.22,...
1055.0,4.3,2.6,1.9,1.23,...
1060.0,4.1,2.7,2.0,1.24,...
...
```

**Interpretation**:
- Row 1: Stressed values at first stressed time step
- Row 2: Stressed values at second stressed time step
- Etc.

---

## Running Your First Scenario

### Scenario: Generate 10 Unconditional Forecasts

No stress conditioning—just pure generative sampling from a trained model.

```powershell
python scripts/run_handler_real.py `
  --data-hist data/germany_macro_augemented.csv `
  --seq-length 24 `
  --feature-size 11 `
  --num-samples 10 `
  --milestone "10" `
  --results-folder results/unconditional_test
```

**What this does**:
1. Loads historical data and computes relative changes
2. Resumes from `results/unconditional_test_24/checkpoint-10.pt`
3. Generates 10 forecast sequences (24 steps each, 11 features)
4. Saves outputs to `results/unconditional_test/`

**Outputs**:
- `samples.npy`: Relative change values
- `samples_absolute.npy`: Absolute values (reconstructed)
- `sample_0.csv`, `sample_1.csv`, ..., `sample_9.csv`: Individual scenario CSVs

---

### Scenario: Generate Conditional Stress Scenarios

Apply specific stress values to selected features at chosen time steps.

```powershell
python scripts/run_handler_real.py `
  --data-hist data/germany_macro_augemented.csv `
  --data-stress data/germany_stress_boom.csv `
  --seq-length 24 `
  --feature-size 11 `
  --num-samples 200 `
  --stressed-features "0,1,2" `
  --stressed-seq-indices "12,13,14,15,16,17,18,19,20,21,22,23" `
  --milestone "10" `
  --results-folder results/stress_boom
```

**What this does**:
1. Loads historical data and stress scenario
2. Conditions generation on features 0, 1, 2 (e.g., GDP, Unemployment, Inflation) at time steps 12–23
3. Generates 200 scenarios that respect the stress constraints
4. Saves to `results/stress_boom/`

**Key Parameters Explained**:
- `--stressed-features "0,1,2"`: Indices of columns in CSV (0-based). Feature 0 = first column (GDP), Feature 1 = second column (Unemployment), etc.
- `--stressed-seq-indices "12,13,...,23"`: Time indices within the 24-step forecast. Here, the last 12 steps are stressed.

---

### Visualizing Results

```powershell
python scripts/plot_generated.py `
  --samples-path results/stress_boom/samples_absolute.npy `
  --hist-csv data/germany_macro_augemented.csv `
  --history-points 24 `
  --overlap 6 `
  --limit-samples 10 `
  --use-absolute `
  --output-pdf results/stress_boom/samples_plot.pdf
```

**Output**: `samples_plot.pdf` with one page per feature showing:
- **Green dashed line**: Last 24 historical data points
- **Red solid lines**: 10 generated scenarios (with transparency)
- **Overlap**: 6 time steps where history and forecast connect

**Open the PDF** to inspect whether scenarios are realistic and respect stress conditions.

---

## Advanced Usage

### Training a Model from Scratch

If you don't have a checkpoint, omit `--milestone`:

```powershell
python scripts/run_handler_real.py `
  --data-hist data/my_historical_data.csv `
  --seq-length 24 `
  --feature-size 7 `
  --num-samples 100 `
  --results-folder results/my_training
```

**What happens**:
1. Model trains for `max_epochs` (set in `diffusion_config.yaml`, default 10)
2. Checkpoints saved every `save_cycle` epochs (default 2) to `results/my_training_24/checkpoint-{epoch}.pt`
3. After training completes, generates 100 samples

**Time estimate**: ~10–30 minutes per epoch on GPU (depends on data size and model complexity)

---

### Customizing Model Architecture

Edit `src/configs/diffusion_configs/diffusion_config.yaml`:

```yaml
model:
  params:
    seq_length: 24          # Forecast horizon
    feature_size: 11        # Number of variables
    n_layer_enc: 3          # Encoder layers (increase for complexity)
    n_layer_dec: 3          # Decoder layers
    d_model: 96             # Hidden dimension (higher = more capacity)
    n_heads: 4              # Attention heads
    timesteps: 250          # Diffusion steps during training
```

**Tips**:
- Increase `d_model` for more expressive power (but slower training)
- Increase `n_layer_enc/dec` for deeper models (better long-range dependencies)
- Adjust `timesteps` to balance quality vs. speed

Then run with:
```powershell
--diffusion-config-path src/configs/diffusion_configs/diffusion_config.yaml
```

---

### Batch Processing: Generate Multiple Stress Scenarios

Create multiple stress CSVs (e.g., `stress_boom.csv`, `stress_recession.csv`, `stress_crisis.csv`) and run:

```powershell
# Boom scenario
python scripts/run_handler_real.py --data-stress data/stress_boom.csv --results-folder results/boom ...

# Recession scenario
python scripts/run_handler_real.py --data-stress data/stress_recession.csv --results-folder results/recession ...

# Crisis scenario
python scripts/run_handler_real.py --data-stress data/stress_crisis.csv --results-folder results/crisis ...
```

Compare PDFs side-by-side to understand scenario differences.

---

### Adjusting Sampling Quality vs. Speed

**Faster (lower quality)**:
```powershell
--sampling-steps 100
```

**Slower (higher quality)**:
```powershell
--sampling-steps 500
```

**Default**: 200 steps (good balance)

---

## Interpreting Results

### Understanding `samples.npy` vs. `samples_absolute.npy`

| File | Content | Use Case |
|------|---------|----------|
| `samples.npy` | Relative changes (%) | Analyze volatility, compare to historical relative changes |
| `samples_absolute.npy` | Absolute values | Plot against historical levels, downstream modeling |

**Example**: If GDP in `samples.npy` is `0.03` at time step 5, it means a **+3% change** from the previous step.

### Reading Individual Sample CSVs

Each `sample_{i}.csv` contains absolute values:
- **Rows**: Time steps (0 to `seq_length - 1`)
- **Columns**: Features (same order as historical CSV)

Load in Excel/Python for further analysis:
```python
import pandas as pd
sample = pd.read_csv("results/stress_boom/sample_0.csv", header=None)
print(sample.head())
```

### Validating Stress Conditioning

**Check if stressed features match expected values**:

1. Load stress CSV and sample CSV
2. Compare columns corresponding to `stressed_features` at `stressed_seq_indices`
3. Values should be very close (within numerical precision)

**Python example**:
```python
import numpy as np
stress = np.loadtxt("data/germany_stress_boom.csv", delimiter=",", skiprows=1)
sample = np.loadtxt("results/stress_boom/sample_0.csv", delimiter=",")

stressed_feats = [0, 1, 2]
stressed_steps = list(range(12, 24))

for step in stressed_steps:
    for feat in stressed_feats:
        print(f"Step {step}, Feat {feat}: Stress={stress[step-12, feat]:.4f}, Sample={sample[step, feat]:.4f}")
```

Expected: Values should match closely.

---

## Common Workflows

### Workflow 1: Evaluate Multiple Checkpoints

Compare model quality at different training stages:

```powershell
# Generate with checkpoint 5
python scripts/run_handler_real.py --milestone "5" --results-folder results/ckpt5 ...

# Generate with checkpoint 10
python scripts/run_handler_real.py --milestone "10" --results-folder results/ckpt10 ...

# Plot both
python scripts/plot_generated.py --samples-path results/ckpt5/samples_absolute.npy --output-pdf results/ckpt5_plot.pdf ...
python scripts/plot_generated.py --samples-path results/ckpt10/samples_absolute.npy --output-pdf results/ckpt10_plot.pdf ...
```

Visual inspection: Which checkpoint produces more realistic scenarios?

---

### Workflow 2: Sensitivity Analysis on Stressed Features

**Test 1**: Stress only GDP (feature 0)
```powershell
--stressed-features "0" --stressed-seq-indices "12,13,...,23"
```

**Test 2**: Stress GDP + Unemployment (features 0, 1)
```powershell
--stressed-features "0,1" --stressed-seq-indices "12,13,...,23"
```

**Test 3**: Stress all features (0 to 10)
```powershell
--stressed-features "0,1,2,3,4,5,6,7,8,9,10" --stressed-seq-indices "12,13,...,23"
```

Compare PDFs: How does adding more constraints affect scenario diversity?

---

### Workflow 3: Generate Baseline + Stress Comparison

**Baseline (no stress)**:
```powershell
python scripts/run_handler_real.py --data-hist data/germany_macro.csv --results-folder results/baseline ...
```

**Stress (boom scenario)**:
```powershell
python scripts/run_handler_real.py --data-hist data/germany_macro.csv --data-stress data/stress_boom.csv --results-folder results/stress ...
```

**Overlay plots**: Use external tool (e.g., Python script) to plot both on same axes.

---

## FAQ

### Q: What if I get "CUDA out of memory"?

**A**: Reduce batch size:
```powershell
--batch-size 16
```
Or switch to CPU (slower):
```python
# In diffusion_ts_adapter.py, force CPU:
device = torch.device('cpu')
```

---

### Q: How do I know if my checkpoint is good?

**A**: Visual inspection via plots + quantitative metrics:
1. **Visual**: Do samples look realistic? Match historical patterns?
2. **Coverage**: Do samples span a reasonable range?
3. **Conditioning**: Do stressed values appear in generated samples?

For formal validation, compute:
- Mean Absolute Error (MAE) on held-out test set
- Distributional metrics (e.g., KL divergence, Wasserstein distance)

---

### Q: Can I use daily data instead of monthly?

**A**: Yes, adjust `seq_length` accordingly. For 2 years of daily data:
```powershell
--seq-length 730  # ~2 years
```

**Note**: Longer sequences require more memory and training time.

---

### Q: How do I add new features to the CSV?

**A**:
1. Add column to historical and stress CSVs
2. Increment `--feature-size`:
   ```powershell
   --feature-size 12  # was 11
   ```
3. Update `feature_names` in plot script if desired:
   ```powershell
   --feature-names "GDP,Unemployment,Inflation,...,NewFeature"
   ```

---

### Q: What if training is too slow?

**A**:
- Use GPU (10-20x faster than CPU)
- Reduce `max_epochs` in `diffusion_config.yaml`
- Reduce data size (use fewer historical rows)
- Lower model complexity (`d_model`, `n_layer_enc/dec`)

---

### Q: Can I use this for non-macroeconomic data?

**A**: Absolutely! The framework is domain-agnostic. Use it for:
- Financial time series (stock prices, returns)
- Energy load forecasting
- Climate data (temperature, precipitation)
- IoT sensor data

Just ensure your CSV follows the required format.

---

### Q: How do I cite this in a report?

**A**:
```
Scenario Generation Framework (2026). 
Diffusion-based stress scenario generation using Diffusion-TS.
https://github.com/yourusername/scenario_generation
```

---

## Next Steps

1. **Explore**: Run multiple scenarios with different stress conditions
2. **Validate**: Compare generated samples to expert-defined scenarios
3. **Integrate**: Use outputs in downstream risk models (e.g., stress testing, portfolio optimization)
4. **Extend**: Add new adapters for alternative generative models (GANs, VAEs)

For architectural details, see **ARCHITECTURE.md**.

For API reference, see **README.md**.

---

**Questions or Issues?**
- Open an issue on GitHub
- Email: your.email@example.com

---

**Last Updated**: January 13, 2026
