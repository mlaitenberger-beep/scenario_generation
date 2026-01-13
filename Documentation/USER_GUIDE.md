control-Shift-V to Run this file 
# Scenario Generation User Guide

Complete step-by-step instructions for generating and visualizing macroeconomic stress scenarios.

---

## Table of Contents

3. [Data Preparation](#data-preparation)
4. [Running Your First Scenario](#running-your-first-scenario)
5. [Advanced Usage](#advanced-usage)
6. [Interpreting Results](#interpreting-results)
7. [Common Workflows](#common-workflows)

---


### Step 1: Navigate to Project Folder


### Step 2: Install Python Dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib pyyaml einops ema-pytorch tqdm
```

**For GPU support** (NVIDIA CUDA):
```bash
# First check your CUDA version
nvidia-smi  # Look for "CUDA Version: 11.8" or similar

# Install PyTorch with matching CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
# Or visit: https://pytorch.org/get-started/locally/
```

### Step 3: Verify Installation and GPU

```bash
python scripts/check_gpu.py
```

Expected output:
```
============================================================
DEPENDENCY & GPU VERIFICATION
============================================================

[1/6] Checking core dependencies...
  ✓ NumPy 1.24.3
  ✓ Pandas 2.0.2
  ✓ scikit-learn 1.3.0
  ✓ Matplotlib 3.7.1
  ✓ PyYAML
  ✓ tqdm 4.65.0

[2/6] Checking PyTorch...
  ✓ PyTorch 2.1.0

[3/6] Checking Diffusion-TS dependencies...
  ✓ einops 0.7.0
  ✓ ema-pytorch

[4/6] Checking CUDA/GPU availability...
  CUDA available: True
  CUDA version: 11.8
  Number of GPUs: 1
  
  GPU 0:
    Name: Tesla V100-SXM2-16GB
    Memory: 16.00 GB
    Compute capability: 7.0

[5/6] Testing GPU tensor operations...
  ✓ Matrix multiplication on GPU successful
  ✓ GPU memory allocated: 7.63 MB

============================================================
SUMMARY
============================================================
✓ All dependencies installed
✓ GPU acceleration available: 1 GPU(s)
✓ Primary device: cuda:0 (Tesla V100-SXM2-16GB)

You can run training with GPU acceleration!
============================================================
```

**If you see "No CUDA GPUs detected"**: The code will still work but will use CPU (much slower).

---

## Data Preparation

### Historical Data Format

Your historical CSV must have:
- **First row**: Feature names (header)
- **Subsequent rows**: Time series values (one row per time step)
- **Columns**: One per feature 



**Requirements**:
- All values must be numeric (float or int)
- No missing values (NaN)
- At least `seq_length + 50` rows recommended for training

### Stress Scenario Format

Your stress CSV defines the conditional values you want to impose:
- **Same format** as historical CSV
- **Length**: Typically equals `seq_length` or the stressed portion
- **Values**: Absolute values (not relative changes)


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
