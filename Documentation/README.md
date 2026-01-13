# Scenario Generation with Diffusion-TS

A modular framework for generating macroeconomic stress scenarios using diffusion-based time-series models. This project provides an end-to-end pipeline from raw CSV data to conditional scenario generation with stress conditioning.

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ (with CUDA support recommended)
- Git (for cloning)

### Installation

1. **Clone the repository** (or navigate to the project folder):
   ```bash
   cd scenario_generation
   ```

2. **Install dependencies**:
   ```bash
   pip install torch numpy pandas scikit-learn matplotlib pyyaml
   ```

3. **Verify installation**:
   ```bash
   python scripts/check_imports_deps.py
   ```

### Basic Usage

**Generate scenarios from trained checkpoint**:

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
  --results-folder results/germany_boom
```

**Visualize results**:

```powershell
python scripts/plot_generated.py `
  --samples-path results/germany_boom/samples_absolute.npy `
  --hist-csv data/germany_macro_augemented.csv `
  --limit-samples 10 `
  --use-absolute `
  --output-pdf results/germany_boom/samples_plot.pdf
```

---

## Project Structure

```
scenario_generation/
├── src/
│   ├── handler.py                    # High-level orchestrator
│   ├── adapters/
│   │   ├── modelAdapter.py           # Abstract adapter interface
│   │   └── diffusion_ts_adapter.py   # Diffusion-TS implementation
│   ├── data/
│   │   ├── inputData.py              # Data preprocessing pipeline
│   │   ├── utils.py                  # Dataset classes
│   │   └── visualizer.py             # (Future) Visualization utilities
│   ├── configs/
│   │   ├── config.yaml               # Global runtime config
│   │   └── diffusion_configs/
│   │       └── diffusion_config.yaml # Model architecture config
│   └── third_party/
│       └── DiffusionTS/              # External diffusion model
├── scripts/
│   ├── run_handler_real.py           # CLI for scenario generation
│   ├── plot_generated.py             # Visualization script
│   └── check_imports_deps.py         # Dependency checker
├── data/
│   ├── germany_macro_augemented.csv  # Historical data
│   └── germany_stress_boom.csv       # Stress scenario values
├── results/                          # Output folder for generated scenarios
├── tests/                            # Unit and integration tests
├── ARCHITECTURE.md                   # Detailed architecture documentation
└── README.md                         # This file
```

---

## Workflow Overview

### 1. Data Preparation

**Input Format**: CSV files with headers, one row per time step, one column per feature.

**Historical Data** (`data_hist`):
```csv
GDP,Unemployment,Inflation,...
1000.5,5.2,2.1,...
1015.3,5.1,2.3,...
...
```

**Stress Scenario** (`data_stress`):
```csv
GDP,Unemployment,Inflation,...
1020.0,6.5,3.0,...
1025.0,7.0,3.5,...
...
```

### 2. Configuration

**Key Parameters**:
- `seq_length`: Forecast horizon (e.g., 24 months)
- `feature_size`: Number of macroeconomic variables (e.g., 11)
- `num_samples`: Number of scenarios to generate (e.g., 200)
- `stressed_features`: Indices of features to condition on (e.g., `[0,1,2]` for GDP, Unemployment, Inflation)
- `stressed_seq_indices`: Time indices where stress values are applied (e.g., last 12 steps: `[12,...,23]`)
- `milestone`: Checkpoint number to resume from (e.g., `"10"` for `checkpoint-10.pt`)

### 3. Generation Process

1. **Data Input**:
   - Load historical and stress CSVs
   - Convert to relative changes: `(new - old) / old`
   - Normalize to [-1, 1] using MinMaxScaler
   - Create overlapping sequences for training context

2. **Model Loading**:
   - Instantiate Diffusion-TS model from config
   - Load checkpoint if provided (or train from scratch)

3. **Conditional Sampling**:
   - Prepare forecast sequence with stress values at specified indices
   - Generate `num_samples` scenarios via diffusion sampling
   - Stress features guide generation through masked attention

4. **Denormalization**:
   - Convert [-1, 1] → [0, 1] → relative changes
   - Apply cumulative product to reconstruct absolute values
   - Save both formats: `samples.npy` (relative), `samples_absolute.npy` (absolute)

### 4. Visualization

- Plot historical tail + generated forecasts per feature
- Green dashed line: historical data
- Red solid lines: generated scenarios (with transparency for overlapping)
- Output: Multi-page PDF (`samples_plot.pdf`)

---

## Command-Line Reference

### `run_handler_real.py`

Generate scenarios from historical and stress data.

**Required Arguments**:
```
--data-hist PATH          Path to historical CSV
--seq-length INT          Forecast sequence length
--feature-size INT        Number of features
```

**Optional Arguments**:
```
--data-stress PATH        Path to stress CSV (for conditional generation)
--num-samples INT         Number of scenarios to generate (default: 100)
--sampling-steps INT      Diffusion sampling steps (default: 200)
--batch-size INT          Training batch size (default: 64)
--milestone STR           Checkpoint number to resume from (e.g., "10")
--results-folder PATH     Output directory (default: ./results/run)
--stressed-features LIST  Comma-separated feature indices (e.g., "0,1,2")
--stressed-seq-indices LIST  Comma-separated time indices (e.g., "12,13,...,23")
--len-historie INT        Number of historical steps to include in forecast input
--diffusion-config-path PATH  Override default diffusion_config.yaml
--no-save                 Do not save outputs (dry run)
```

**Example (Stress Scenario)**:
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
  --results-folder results/germany_boom
```

**Outputs**:
- `samples.npy`: Relative change values (shape: `(num_samples, seq_length, feature_size)`)
- `samples_absolute.npy`: Absolute values reconstructed from relative changes
- `sample_{i}.csv`: Individual scenario CSVs (absolute values)

---

### `plot_generated.py`

Visualize generated scenarios against historical data.

**Required Arguments**:
```
--samples-path PATH       Path to samples.npy or samples_absolute.npy
--hist-csv PATH           Path to historical CSV
```

**Optional Arguments**:
```
--history-points INT      Number of historical points to plot (default: 24)
--overlap INT             Overlap between history and forecast (default: 6)
--limit-samples INT       Max scenarios to plot (default: 10)
--feature-names LIST      Comma-separated feature names (default: CSV headers)
--output-pdf PATH         Output PDF path (default: samples_plot.pdf)
--use-absolute            Treat samples as absolute values (skip conversion)
```

**Example**:
```powershell
python scripts/plot_generated.py `
  --samples-path results/germany_boom/samples_absolute.npy `
  --hist-csv data/germany_macro_augemented.csv `
  --history-points 24 `
  --overlap 6 `
  --limit-samples 10 `
  --use-absolute `
  --output-pdf results/germany_boom/samples_plot.pdf
```

---

## Configuration Files

### `src/configs/config.yaml`

Global runtime defaults. Overridden by CLI arguments.

```yaml
seq_length: 24
feature_size: 11
num_samples: 200
stressed_features: [0, 1, 2]
stressed_seq_indices: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
len_historie: 6
milestone: "10"
batch_size: 64
sampling_steps: 200
results_folder: results/my_run
data_hist_path: data/germany_macro_augemented.csv
data_stress_path: data/germany_stress_boom.csv
```

### `src/configs/diffusion_configs/diffusion_config.yaml`

Diffusion-TS model architecture and training hyperparameters.

**Key Sections**:
- `model.params`: Architecture (layers, d_model, heads, timesteps)
- `solver`: Training (learning rate, epochs, save frequency)
- `dataloader`: Dataset configuration

**Note**: Runtime config dynamically overrides these values via `create_model_config()`.

---

## Training from Scratch

If no checkpoint is available, omit `--milestone`:

```powershell
python scripts/run_handler_real.py `
  --data-hist data/germany_macro_augemented.csv `
  --seq-length 24 `
  --feature-size 11 `
  --num-samples 100 `
  --results-folder results/new_training
```

The model will train for `max_epochs` (defined in `diffusion_config.yaml`) and save checkpoints every `save_cycle` epochs to `results/new_training_{seq_length}/checkpoint-{epoch}.pt`.

---

## Stress Conditioning

### How It Works

1. **Specify Stressed Features**: Choose which variables to condition on (e.g., GDP, Unemployment, Inflation → indices `[0,1,2]`)
2. **Specify Stress Indices**: Define which time steps have known stress values (e.g., last 12 steps → `[12,...,23]`)
3. **Masked Attention**: During sampling, the diffusion model's attention mechanism is masked to respect the known stressed values while generating the rest of the sequence

### Example: Boom Scenario

**Stress CSV** (`germany_stress_boom.csv`):
```csv
GDP,Unemployment,Inflation,...
1050.0,4.5,2.5,...  # t=12: boom starts
1055.0,4.3,2.6,...  # t=13
...
1100.0,4.0,2.8,...  # t=23
```

**Command**:
```powershell
--data-stress data/germany_stress_boom.csv
--stressed-features "0,1,2"  # GDP, Unemployment, Inflation
--stressed-seq-indices "12,13,14,15,16,17,18,19,20,21,22,23"  # Last 12 steps
```

**Result**: Generated scenarios will match the stressed GDP/Unemployment/Inflation values at steps 12-23, while other features and earlier time steps are sampled freely (conditioned on historical context).

---

## Troubleshooting

### Issue: `FileNotFoundError: checkpoint-10.pt not found`

**Cause**: Trainer appends `_{seq_length}` to `results_folder`. If checkpoint is in `results/germany_boom_24/` but you pass `--results-folder results/germany_boom_24`, it looks for `results/germany_boom_24_24/`.

**Solution**: Pass the base folder name without the suffix:
```powershell
--results-folder results/germany_boom
```

### Issue: Samples are all near zero

**Cause**: Plotting relative changes instead of absolute values.

**Solution**: Use `samples_absolute.npy` with `--use-absolute` flag:
```powershell
--samples-path results/germany_boom/samples_absolute.npy --use-absolute
```

### Issue: `stressed_seq_indices required when data_stress is provided`

**Cause**: Stress conditioning requires both `--stressed-features` and `--stressed-seq-indices`.

**Solution**: Add both arguments:
```powershell
--stressed-features "0,1,2" --stressed-seq-indices "12,13,...,23"
```

### Issue: CUDA out of memory

**Cause**: Batch size too large for GPU memory.

**Solution**: Reduce batch size:
```powershell
--batch-size 32  # or lower
```

---

## Testing

Run unit tests:
```bash
pytest tests/
```

Run integration test (quick smoke test):
```bash
python src/tests/test_handler_realdata_quick.py
```

---

## Performance Tips

1. **Use GPU**: Ensure PyTorch detects CUDA: `torch.cuda.is_available()`
2. **Resume from checkpoint**: Training is expensive; always save checkpoints and resume
3. **Adjust sampling steps**: Lower `--sampling-steps` (e.g., 100 instead of 200) for faster inference at slight quality cost
4. **Limit samples for debugging**: Use `--limit-samples 5` during development

---

## Extending the Framework

### Add a New Generative Model

1. Create `src/adapters/new_model_adapter.py` implementing the adapter interface
2. Register in `ModelAdapter.__init__()`:
   ```python
   if self.model == "new_model":
       from .new_model_adapter import NewModelAdapter
       self.adapter = NewModelAdapter(config)
   ```
3. Use: `Handler(model='new_model', config={...})`

See `ARCHITECTURE.md` for detailed extension guide.

---

## Key Concepts

### Relative Changes
Data is transformed to percentage changes (`(new - old) / old`) to ensure:
- Scale invariance across features
- Better stationarity for model training
- Interpretable stress scenarios (% shocks)

### Normalization
- **MinMaxScaler**: Fits on historical relative changes, stores min/max per feature
- **[-1, 1] Mapping**: Required by Diffusion-TS model input
- **Invertible**: Denormalization reconstructs original scale exactly

### Overlapping Sequences
Training dataset is created by sliding a window of length `seq_length` over historical data, increasing sample count and providing temporal context.

### Cumulative Product
Relative changes are converted to absolute values via:
```python
current = last_historical_value
for t in range(seq_length):
    current = current * (1 + rel_change[t])
    absolute[t] = current
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Diffusion model backend |
| `numpy` | Array operations |
| `pandas` | CSV I/O |
| `scikit-learn` | MinMaxScaler normalization |
| `matplotlib` | Plotting |
| `pyyaml` | Config file parsing |

Install all:
```bash
pip install torch numpy pandas scikit-learn matplotlib pyyaml
```

---

## Citation

If you use this framework in research, please cite:

```bibtex
@software{scenario_generation_2026,
  title={Scenario Generation with Diffusion-TS},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/scenario_generation}
}
```

---

## License

[Specify license, e.g., MIT, Apache 2.0]

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Last Updated**: January 13, 2026
