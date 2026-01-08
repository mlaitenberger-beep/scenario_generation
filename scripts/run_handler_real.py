"""Run the Handler end-to-end with real data.

This script allows you to plug in your real historical and stress CSV files and run
the Handler -> ModelAdapter -> Diffusion-TS flow. It avoids adding any heavy
test harness and instead runs the actual adapter; you can monkeypatch or set
config values via CLI if you need to override behavior.

Example:
  python scripts/run_handler_real.py --data-hist data/hist.csv --seq-length 24 --feature-size 3 --num-samples 100 --results-folder ./results/run1

Notes:
 - The script adds the project's `src` folder to PYTHONPATH so imports succeed
 - The Diffusion-TS adapter will lazily import torch and its internals when needed
 - Output samples are saved under the specified results folder as NumPy .npy files
"""

import argparse
import os
import sys
import traceback
import json

# ensure src is on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from handler import Handler
import numpy as np


def parse_int_list(s):
    if s is None or s == '':
        return None
    return [int(x) for x in s.split(',') if x.strip()]


def main():
    p = argparse.ArgumentParser(description='Run Handler with real data to generate scenarios')
    p.add_argument('--data-hist', required=True, help='Path to historical data CSV')
    p.add_argument('--data-stress', default=None, help='Path to stress data CSV (optional)')
    p.add_argument('--seq-length', type=int, required=True, help='Sequence length to use')
    p.add_argument('--feature-size', type=int, required=True, help='Number of features in the data')
    p.add_argument('--num-samples', type=int, default=100, help='Number of samples to generate')
    p.add_argument('--sampling-steps', type=int, default=200, help='Sampling steps for generation')
    p.add_argument('--diffusion-config-path', default=None, help='Optional path to diffusion config YAML')
    p.add_argument('--results-folder', default='./results/run', help='Folder where outputs will be saved')
    p.add_argument('--batch-size', type=int, default=64, help='Batch size used for training/loading')
    p.add_argument('--milestone', default=None, help='Milestone to resume training from (optional)')
    p.add_argument('--stressed-features', default=None, help='Comma-separated list of stressed feature indices (e.g., "0,1")')
    p.add_argument('--stressed-seq-indices', default=None, help='Comma-separated list of indices in forecast sequence to stress (required if --stressed-features is set)')
    p.add_argument('--len-historie', type=int, default=None, help='Length of history to use when preparing forecast sequence')
    p.add_argument('--no-save', action='store_true', help='Do not save samples; just run and print shapes')
    args = p.parse_args()

    # Validate: if stressed-features is set, stressed-seq-indices must also be set
    if args.stressed_features is not None and args.stressed_seq_indices is None:
        p.error('--stressed-seq-indices is required when --stressed-features is provided')
    if args.stressed_seq_indices is not None and args.stressed_features is None:
        p.error('--stressed-features is required when --stressed-seq-indices is provided')

    config = {
        'data_hist_path': args.data_hist,
        'data_stress_path': args.data_stress,
        'seq_length': args.seq_length,
        'feature_size': args.feature_size,
        'num_samples': args.num_samples,
        'sampling_steps': args.sampling_steps,
        'diffusion_config_path': args.diffusion_config_path,
        'results_folder': args.results_folder,
        'batch_size': args.batch_size,
        'milestone': args.milestone,
        'stressed_features': parse_int_list(args.stressed_features),
        'stressed_seq_indices': parse_int_list(args.stressed_seq_indices),
        'len_historie': args.len_historie,
    }

    os.makedirs(args.results_folder, exist_ok=True)

    print('Running Handler with config:')
    print(json.dumps({k: v for k, v in config.items() if v is not None}, indent=2))

    try:
        handler = Handler('diffusion_ts', config)
        samples = handler.createScenarios()

        if samples is None:
            print('Handler returned no samples')
            return 1

        print('Samples shape:', getattr(samples, 'shape', 'unknown'))

        if not args.no_save:
            npy_path = os.path.join(args.results_folder, 'samples.npy')
            np.save(npy_path, samples)
            print('Saved samples to', npy_path)

            # Optionally save each sample to CSV for inspection
            try:
                samples_arr = np.asarray(samples)
                n = samples_arr.shape[0]
                for i in range(n):
                    csv_path = os.path.join(args.results_folder, f'sample_{i}.csv')
                    # flatten last two dims: (seq_len, feat) -> columns
                    np.savetxt(csv_path, samples_arr[i].reshape(samples_arr.shape[1], samples_arr.shape[2]), delimiter=',')
                print(f'Saved {n} sample CSVs to {args.results_folder}')
            except Exception as e:
                print('Failed to save individual CSVs:', e)

        return 0
    except Exception as e:
        print('Error running Handler:')
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
