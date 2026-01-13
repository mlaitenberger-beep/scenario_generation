control-Shift-V to Run this file 
# Scenario Generation Architecture

## Overview

This project generates macroeconomic stress scenarios using a diffusion-based time-series model (Diffusion-TS). The architecture follows a modular adapter pattern that separates business logic from model-specific implementations, allowing easy integration of alternative generative models in the future.

## High-Level Flow

```
User Request
    ↓
Handler (orchestrator)
    ↓
ModelAdapter (abstraction layer)
    ↓
Diffusion_ts_adapter (model-specific implementation)
    ↓
Diffusion-TS Third-Party Model
    ↓
Generated Scenarios (denormalized outputs)
```

---

## Component Responsibilities

### 1. **Handler** (`src/handler.py`)

**Purpose**: High-level orchestrator that coordinates the entire scenario generation workflow.

**Key Responsibilities**:
- Accepts user configuration (data paths, model parameters, stress conditions)
- Delegates model-specific operations to `ModelAdapter`
- Returns denormalized scenario samples ready for downstream use

**Public Interface**:
```python
handler = Handler(model='diffusion_ts', config={...})
scenarios = handler.createScenarios()
# Returns: numpy array (num_samples, seq_length, num_features)
```

**Workflow Steps**:
1. `create_model_config()` - Merge user config with model defaults
2. `load_model()` - Prepare data and instantiate model/trainer
3. `train()` - Train from scratch or resume from checkpoint
4. `predict()` - Generate forecast samples
5. `data_output()` - Denormalize samples back to original scale

---

### 2. **ModelAdapter** (`src/adapters/modelAdapter.py`)

**Purpose**: Abstract interface between `Handler` and model-specific implementations. Provides a unified API regardless of the underlying generative model.

**Key Responsibilities**:
- Factory pattern: instantiates the correct adapter based on `model` parameter (e.g., `"diffusion_ts"`)
- Standardizes method signatures across all model types
- Manages state (trainer, samples, scaler) shared across workflow steps

**Supported Models**:
- `"diffusion_ts"` → `Diffusion_ts_adapter`
- _(extensible: add new models by creating additional adapters)_

**Public Methods**:
| Method | Description |
|--------|-------------|
| `create_model_config()` | Build/merge configuration for the model |
| `load_model()` | Prepare input data and instantiate model |
| `train()` | Train model or resume from checkpoint |
| `predict()` | Generate forecast samples |
| `data_output()` | Denormalize samples to original scale |

---

### 3. **Diffusion_ts_adapter** (`src/adapters/diffusion_ts_adapter.py`)

**Purpose**: Model-specific implementation for the Diffusion-TS generative model. Handles data preprocessing, training, and inference using the third-party Diffusion-TS engine.

**Key Responsibilities**:
- **Lazy imports**: Avoids loading heavy dependencies (torch, CUDA) until methods are called
- **Data pipeline**: 
  - Reads historical and stress CSVs via `InputData`
  - Converts to relative changes (percentage changes between time steps)
  - Normalizes to [-1, 1] range using MinMaxScaler
  - Creates overlapping sequences and stress-conditioned forecast inputs
- **Model management**: Instantiates Diffusion-TS model and Trainer
- **Training**: Trains from scratch or resumes from checkpoint milestone
- **Inference**: Generates samples via conditional sampling (stress features masked)
- **Denormalization**: Converts model outputs back to relative-change scale

**Configuration**:
- Uses `diffusion_config.yaml` for model architecture (layers, d_model, timesteps, etc.)
- Merges runtime config (seq_length, feature_size, results_folder) into YAML

**Data Flow**:
```
CSV (absolute values)
    ↓ relative_change()
Relative changes (% changes)
    ↓ normalize() → MinMaxScaler + [-1,1] mapping
Normalized sequences
    ↓ create_overlapping_sequences()
Training dataset
    ↓ Trainer.train() or Trainer.load(milestone)
Trained model
    ↓ Trainer.restore(forecast_dataloader)
Generated samples (normalized)
    ↓ data_output() → inverse MinMaxScaler + [0,1]→[-1,1]
Denormalized relative changes
```

**Stress Conditioning**:
- Accepts `stressed_features` (list of feature indices, e.g., `[0, 1, 2]` for GDP, Unemployment, Inflation)
- Accepts `stressed_seq_indices` (list of time indices where stress values are known, e.g., `[12,13,...,23]`)
- During inference, the model's attention mechanism is conditioned on these known values to generate coherent scenarios

---

### 4. **InputData** (`src/data/inputData.py`)

**Purpose**: Handles all data preprocessing steps—converting raw CSV time series into model-ready normalized sequences.

**Key Responsibilities**:
- Read historical and stress CSV files (with headers)
- Compute relative changes: `(new - old) / old`
- Normalize using MinMaxScaler fitted on historical data
- Map to [-1, 1] range for model input
- Create overlapping sequences for training
- Prepare forecast sequence with stress values inserted at specified indices

**Important Methods**:
| Method | Description |
|--------|-------------|
| `relative_change()` | Convert absolute values → percentage changes |
| `normalize()` | Fit MinMaxScaler on historical data and transform both hist/stress |
| `create_overlapping_sequences()` | Sliding window over historical data |
| `prepare_forcast_seq()` | Build forecast input with stress conditioning |

**Stress Handling**:
- Stress CSV first row is computed as relative change from last historical value
- Subsequent stress rows are relative to previous stress values
- This ensures continuity between historical and stress regimes

---

### 5. **Third-Party: Diffusion-TS** (`src/third_party/DiffusionTS/`)

**Purpose**: External diffusion-based time-series generation model (research code).

**Key Components**:
- `engine/solver.py`: `Trainer` class for training/inference
- `Models/interpretable_diffusion/`: Diffusion model architecture
- `Utils/`: Data utilities, I/O, metrics
- `Data/build_dataloader.py`: Dataset construction

**Integration Points**:
- `Trainer.train()` - Trains model for specified epochs
- `Trainer.load(milestone)` - Resumes from checkpoint (e.g., `checkpoint-10.pt`)
- `Trainer.restore(dataloader, shape, sampling_steps)` - Generates samples via conditional sampling

**Note**: The Trainer automatically appends `_{seq_length}` to `results_folder` (e.g., `germany_boom` → `germany_boom_24` for 24-step sequences).

---

## Data Flow Example

### Input Configuration
```python
config = {
    'data_hist_path': 'data/germany_macro_augemented.csv',
    'data_stress_path': 'data/germany_stress_boom.csv',
    'seq_length': 24,
    'feature_size': 11,
    'num_samples': 200,
    'stressed_features': [0, 1, 2],  # GDP, Unemployment, Inflation
    'stressed_seq_indices': [12, 13, ..., 23],  # Last 12 steps stressed
    'milestone': '10',  # Resume from checkpoint-10.pt
    'results_folder': 'results/germany_boom'
}
```

### Execution Steps

1. **Handler.createScenarios()**
   - Calls `ModelAdapter.create_model_config()` → merges config with YAML defaults
   
2. **ModelAdapter.load_model()**
   - Calls `Diffusion_ts_adapter.data_input()`:
     - Reads CSV files
     - Computes relative changes
     - Normalizes to [-1, 1]
     - Creates overlapping sequences
     - Prepares forecast sequence with stress values
   - Instantiates Diffusion-TS model and Trainer

3. **ModelAdapter.train()**
   - Calls `Diffusion_ts_adapter.train(trainer)`:
     - If `milestone` provided: `Trainer.load(milestone='10')` → resumes from `results/germany_boom_24/checkpoint-10.pt`
     - Otherwise: `Trainer.train()` → trains from scratch

4. **ModelAdapter.predict()**
   - Calls `Diffusion_ts_adapter.predict(trainer)`:
     - Replicates forecast sequence `num_samples` times
     - Creates `CustomDataset` with stress masking
     - Calls `Trainer.restore(...)` → generates 200 samples conditioned on stressed features
     - Returns samples in normalized space (shape: `(200, 24, 11)`)

5. **ModelAdapter.data_output()**
   - Calls `Diffusion_ts_adapter.data_output(samples)`:
     - Converts [-1, 1] → [0, 1] via `(x + 1) / 2`
     - Applies `MinMaxScaler.inverse_transform()` to get relative changes
     - Returns denormalized samples (shape: `(200, 24, 11)`)

6. **Post-Processing** (in `run_handler_real.py`)
   - Converts relative changes → absolute values via cumulative product:
     ```python
     current = last_hist_value
     for t in range(seq_len):
         current = current * (1 + rel_change[t])
         absolute_values[t] = current
     ```
   - Saves:
     - `samples.npy` (relative changes)
     - `samples_absolute.npy` (absolute values)
     - Individual CSV files per sample

---

## Configuration Files

### `src/configs/config.yaml`
Global defaults for runtime parameters:
- `seq_length`, `feature_size`, `num_samples`
- `stressed_features`, `stressed_seq_indices`, `len_historie`
- `milestone`, `batch_size`, `sampling_steps`

### `src/configs/diffusion_configs/diffusion_config.yaml`
Diffusion-TS model architecture and training hyperparameters:
- Model: `seq_length`, `feature_size`, `n_layer_enc/dec`, `d_model`, `timesteps`
- Solver: `base_lr`, `max_epochs`, `save_cycle`, `results_folder`
- Dataloader: dataset params, batch_size, sample_size

**Dynamic Override**: Runtime config (from CLI or `config.yaml`) overrides YAML values via `create_model_config()`.

---

## Scripts

### `scripts/run_handler_real.py`
CLI interface for end-to-end scenario generation:
```bash
python scripts/run_handler_real.py \
  --data-hist data/germany_macro.csv \
  --data-stress data/germany_stress_boom.csv \
  --seq-length 24 \
  --feature-size 11 \
  --num-samples 200 \
  --stressed-features "0,1,2" \
  --stressed-seq-indices "12,13,14,...,23" \
  --milestone "10" \
  --results-folder results/germany_boom
```

**Outputs**:
- `samples.npy` (relative changes)
- `samples_absolute.npy` (absolute values)
- `sample_{i}.csv` (per-sample CSV files)

### `scripts/plot_generated.py`
Visualization tool for comparing generated scenarios vs. historical tail:
```bash
python scripts/plot_generated.py \
  --samples-path results/germany_boom/samples_absolute.npy \
  --hist-csv data/germany_macro.csv \
  --history-points 24 \
  --overlap 6 \
  --limit-samples 10 \
  --use-absolute \
  --output-pdf results/germany_boom/samples_plot.pdf
```

**Output**: Multi-page PDF with one page per feature, showing historical tail (green dashed) and generated samples (red solid).

---

## Extension Points

### Adding a New Generative Model

1. **Create Adapter**: Implement `src/adapters/new_model_adapter.py` inheriting from `ModelAdapter`
2. **Implement Methods**:
   - `create_model_config()` → load model-specific config
   - `load_model()` → instantiate model and preprocessing
   - `train()` → train or resume
   - `predict()` → generate samples
   - `data_output()` → denormalize outputs
3. **Register in ModelAdapter**:
   ```python
   if self.model == "new_model":
       from .new_model_adapter import NewModelAdapter
       self.adapter = NewModelAdapter(config)
   ```
4. **Use**: `Handler(model='new_model', config={...})`

### Custom Data Preprocessing

Subclass `InputData` and override `relative_change()`, `normalize()`, or `prepare_forcast_seq()` to implement custom transformations (e.g., log-returns, z-score normalization).

---

## Key Design Decisions

### Why Relative Changes?
- **Scale invariance**: Percentage changes are comparable across features with different units (e.g., GDP in billions vs. interest rates in %)
- **Stationarity**: Relative changes are more stationary than raw levels, improving model training
- **Interpretability**: Stress scenarios naturally expressed as percentage shocks

### Why MinMaxScaler + [-1, 1]?
- **Model compatibility**: Diffusion models typically expect inputs in [-1, 1] or [0, 1]
- **Bounded range**: Prevents extreme outliers from dominating loss functions
- **Invertible**: Scaler stores min/max, allowing exact reconstruction

### Why Lazy Imports?
- **Faster startup**: Avoids loading 2GB+ PyTorch/CUDA libraries during module import
- **Test compatibility**: Tests can import modules without GPU access
- **Modularity**: Heavy dependencies only loaded when specific adapters are used

### Why Overlapping Sequences?
- **Data augmentation**: Increases training samples from limited historical data
- **Temporal context**: Model learns dependencies across different time windows
- **Standard practice**: Common in time-series forecasting (sliding window)

---

## Troubleshooting

### Issue: `FileNotFoundError: checkpoint-10.pt`
**Cause**: Trainer appends `_{seq_length}` to `results_folder`. If you pass `results/germany_boom_24` and `seq_length=24`, it looks for `results/germany_boom_24_24/`.

**Solution**: Pass base folder name without suffix: `--results-folder results/germany_boom`

### Issue: Generated samples are all near zero
**Cause**: Plotting relative changes instead of absolute values.

**Solution**: Use `samples_absolute.npy` with `--use-absolute` flag in plot script.

### Issue: `stressed_seq_indices required when data_stress is provided`
**Cause**: Stress conditioning requires specifying which time indices and features are stressed.

**Solution**: Add `--stressed-features "0,1,2"` and `--stressed-seq-indices "12,13,...,23"` to CLI.

---

## Dependencies

- **Python 3.8+**
- **PyTorch 2.0+** (for Diffusion-TS)
- **NumPy, Pandas, scikit-learn** (data processing)
- **Matplotlib** (plotting)
- **PyYAML** (config loading)

Install via:
```bash
pip install torch numpy pandas scikit-learn matplotlib pyyaml
```

---

## Future Enhancements

1. **Probabilistic metrics**: Add quantile-based validation (e.g., coverage of historical extremes)
2. **Multi-horizon forecasting**: Support variable-length forecasts
3. **Conditional guidance**: Expose diffusion classifier-free guidance strength
4. **Ensemble generation**: Combine multiple checkpoints or models
5. **Real-time inference**: Optimize for low-latency scenario generation
6. **Alternative models**: Integrate GANs, VAEs, or transformer-based generators

---

## References

- **Diffusion-TS Paper**: [Link to original paper if available]
- **MinMaxScaler**: [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- **Diffusion Models**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)

---

**Last Updated**: January 13, 2026
