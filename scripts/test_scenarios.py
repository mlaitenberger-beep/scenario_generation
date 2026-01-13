"""Test script to validate Handler implementation with multiple scenarios.

This script runs various test scenarios to ensure the Handler, ModelAdapter, and 
data processing methods work correctly under different configurations.

Usage:
    python scripts/test_scenarios.py
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

# ensure src is on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from handler import Handler
from data.inputData import InputData


class TestRunner:
    def __init__(self, base_results_folder='./results/test_runs'):
        self.base_results_folder = base_results_folder
        self.results = []
        os.makedirs(base_results_folder, exist_ok=True)
        
    def log(self, message, level='INFO'):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] [{level}] {message}")
        
    def run_test(self, test_name, test_func):
        """Run a single test and capture results."""
        self.log(f"Starting test: {test_name}", 'TEST')
        try:
            test_func()
            self.results.append({'test': test_name, 'status': 'PASSED', 'error': None})
            self.log(f"✓ {test_name} PASSED", 'PASS')
            return True
        except Exception as e:
            self.results.append({'test': test_name, 'status': 'FAILED', 'error': str(e)})
            self.log(f"✗ {test_name} FAILED: {str(e)}", 'FAIL')
            traceback.print_exc()
            return False
    
    def print_summary(self):
        """Print test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r['status'] == 'PASSED')
        failed = total - passed
        
        self.log("=" * 60, 'INFO')
        self.log("TEST SUMMARY", 'INFO')
        self.log("=" * 60, 'INFO')
        self.log(f"Total tests: {total}", 'INFO')
        self.log(f"Passed: {passed}", 'INFO')
        self.log(f"Failed: {failed}", 'INFO')
        
        if failed > 0:
            self.log("\nFailed tests:", 'INFO')
            for r in self.results:
                if r['status'] == 'FAILED':
                    self.log(f"  - {r['test']}: {r['error']}", 'FAIL')


def test_input_data_loading(runner):
    """Test 1: Basic InputData loading and processing."""
    runner.log("Test: InputData loading and normalization")
    
    data_hist = 'scenario_generation/data/germany_macro_augemented.csv'
    data_stress = 'scenario_generation/data/germany_stress_boom.csv'
    
    input_data = InputData(data_hist, data_stress)
    
    # Check data loaded
    assert input_data.data_hist is not None, "Historical data not loaded"
    assert input_data.data_stress is not None, "Stress data not loaded"
    
    # Apply relative change
    input_data.relative_change()
    assert input_data.data_hist_rel is not None, "Relative change not calculated for historical data"
    
    # Normalize
    scaler = input_data.normalize()
    assert scaler is not None, "Scaler not returned"
    assert input_data.data_hist_rel_norm is not None, "Normalized data not created"
    
    runner.log(f"  Data shapes: hist={input_data.data_hist.shape}, stress={input_data.data_stress.shape}")


def test_overlapping_sequences(runner):
    """Test 2: Creating overlapping sequences."""
    runner.log("Test: Creating overlapping sequences")
    
    data_hist = 'scenario_generation/data/germany_macro_augemented.csv'
    input_data = InputData(data_hist, None)
    input_data.relative_change()
    input_data.normalize()
    
    seq_length = 24
    input_data.create_overlapping_sequences(seq_length)
    
    assert input_data.overlapping_sequences is not None, "Overlapping sequences not created"
    assert input_data.overlapping_sequences.shape[1] == seq_length, f"Sequence length mismatch: expected {seq_length}, got {input_data.overlapping_sequences.shape[1]}"
    
    runner.log(f"  Overlapping sequences shape: {input_data.overlapping_sequences.shape}")


def test_forecast_seq_with_stress(runner):
    """Test 3: Preparing forecast sequence with stress."""
    runner.log("Test: Forecast sequence with stress values")
    
    data_hist = 'scenario_generation/data/germany_macro_augemented.csv'
    data_stress = 'scenario_generation/data/germany_stress_boom.csv'
    
    input_data = InputData(data_hist, data_stress)
    input_data.relative_change()
    input_data.normalize()
    input_data.create_overlapping_sequences(24)
    
    stressed_features = [0, 1]
    stressed_seq_indices = [6, 7, 8, 9, 10]
    len_historie = 6
    
    forecast_seq = input_data.prepare_forcast_seq(stressed_features, stressed_seq_indices, len_historie)
    
    assert forecast_seq is not None, "Forecast sequence not created"
    assert forecast_seq.shape[0] == 1, "Forecast sequence batch size should be 1"
    assert forecast_seq.shape[1] == 24, f"Forecast sequence length mismatch: expected 24, got {forecast_seq.shape[1]}"
    
    runner.log(f"  Forecast sequence shape: {forecast_seq.shape}")


def test_forecast_seq_without_stress(runner):
    """Test 4: Preparing forecast sequence without stress."""
    runner.log("Test: Forecast sequence without stress values")
    
    data_hist = 'scenario_generation/data/germany_macro_augemented.csv'
    
    input_data = InputData(data_hist, None)
    input_data.relative_change()
    input_data.normalize()
    input_data.create_overlapping_sequences(24)
    
    len_historie = 12
    forecast_seq = input_data.prepare_forcast_seq(None, None, len_historie)
    
    assert forecast_seq is not None, "Forecast sequence not created"
    assert forecast_seq.shape[1] == 24, f"Forecast sequence length mismatch"
    
    runner.log(f"  Forecast sequence shape: {forecast_seq.shape}")


def test_forecast_seq_with_none_len_historie(runner):
    """Test 5: Forecast sequence with None len_historie (should auto-default)."""
    runner.log("Test: Forecast sequence with None len_historie")
    
    data_hist = 'scenario_generation/data/germany_macro_augemented.csv'
    
    input_data = InputData(data_hist, None)
    input_data.relative_change()
    input_data.normalize()
    input_data.create_overlapping_sequences(24)
    
    # Should not fail with None - should use default
    forecast_seq = input_data.prepare_forcast_seq(None, None, None)
    
    assert forecast_seq is not None, "Forecast sequence not created"
    runner.log(f"  Forecast sequence shape: {forecast_seq.shape}")


def test_handler_basic_config(runner):
    """Test 6: Handler with basic configuration (no training)."""
    runner.log("Test: Handler initialization with basic config")
    
    config = {
        'data_hist_path': 'scenario_generation/data/germany_macro_augemented.csv',
        'seq_length': 24,
        'feature_size': 5,
        'num_samples': 10,
        'sampling_steps': 50,
        'results_folder': os.path.join(runner.base_results_folder, 'test6'),
        'batch_size': 16,
    }
    
    handler = Handler('diffusion_ts', config)
    assert handler is not None, "Handler not created"
    assert handler.modelAdapter is not None, "ModelAdapter not initialized"
    
    runner.log(f"  Handler initialized successfully")


def test_handler_with_stress(runner):
    """Test 7: Handler with stress configuration."""
    runner.log("Test: Handler with stress data")
    
    config = {
        'data_hist_path': 'scenario_generation/data/germany_macro_augemented.csv',
        'data_stress_path': 'scenario_generation/data/germany_stress_boom.csv',
        'seq_length': 24,
        'feature_size': 5,
        'num_samples': 10,
        'sampling_steps': 50,
        'results_folder': os.path.join(runner.base_results_folder, 'test7'),
        'batch_size': 16,
        'stressed_features': [0, 1],
        'stressed_seq_indices': [6, 7, 8, 9, 10],
        'len_historie': 6,
    }
    
    handler = Handler('diffusion_ts', config)
    assert handler is not None, "Handler not created"
    
    runner.log(f"  Handler with stress initialized successfully")


def test_different_seq_lengths(runner):
    """Test 8: Different sequence lengths."""
    runner.log("Test: Different sequence lengths")
    
    for seq_len in [12, 24, 36]:
        data_hist = 'scenario_generation/data/germany_macro_augemented.csv'
        input_data = InputData(data_hist, None)
        input_data.relative_change()
        input_data.normalize()
        input_data.create_overlapping_sequences(seq_len)
        
        assert input_data.overlapping_sequences.shape[1] == seq_len, f"Sequence length {seq_len} failed"
        runner.log(f"  Sequence length {seq_len}: {input_data.overlapping_sequences.shape}")


def test_edge_case_single_stress_index(runner):
    """Test 9: Single stress index (start index mode)."""
    runner.log("Test: Single stress start index")
    
    data_hist = 'scenario_generation/data/germany_macro_augemented.csv'
    data_stress = 'scenario_generation/data/germany_stress_boom.csv'
    
    input_data = InputData(data_hist, data_stress)
    input_data.relative_change()
    input_data.normalize()
    input_data.create_overlapping_sequences(24)
    
    stressed_features = [0, 1]
    stressed_seq_indices = [10]  # Single start index
    len_historie = 6
    
    forecast_seq = input_data.prepare_forcast_seq(stressed_features, stressed_seq_indices, len_historie)
    
    assert forecast_seq is not None, "Forecast sequence not created"
    runner.log(f"  Single stress index mode: {forecast_seq.shape}")


def test_config_validation(runner):
    """Test 10: Config validation (missing required params)."""
    runner.log("Test: Config validation")
    
    data_hist = 'scenario_generation/data/germany_macro_augemented.csv'
    data_stress = 'scenario_generation/data/germany_stress_boom.csv'
    
    input_data = InputData(data_hist, data_stress)
    input_data.relative_change()
    input_data.normalize()
    input_data.create_overlapping_sequences(24)
    
    # Should raise error when stress data provided but stressed_features is None
    try:
        forecast_seq = input_data.prepare_forcast_seq(None, [6, 7, 8], 6)
        raise AssertionError("Should have raised ValueError for missing stressed_features")
    except ValueError as e:
        runner.log(f"  Correctly caught missing stressed_features: {str(e)}")


def test_data_shape_consistency(runner):
    """Test 11: Data shape consistency through pipeline."""
    runner.log("Test: Data shape consistency")
    
    data_hist = 'scenario_generation/data/germany_macro_augemented.csv'
    
    input_data = InputData(data_hist, None)
    original_shape = input_data.data_hist.shape
    
    input_data.relative_change()
    rel_shape = input_data.data_hist_rel.shape
    assert original_shape[1] == rel_shape[1], "Feature count mismatch after relative_change"
    
    input_data.normalize()
    norm_shape = input_data.data_hist_rel_norm.shape
    assert rel_shape == norm_shape, "Shape mismatch after normalization"
    
    input_data.create_overlapping_sequences(24)
    seq_shape = input_data.overlapping_sequences.shape
    assert seq_shape[2] == original_shape[1], "Feature count mismatch in sequences"
    
    runner.log(f"  Shape consistency: original={original_shape}, sequences={seq_shape}")


def test_full_pipeline_with_checkpoint(runner):
    """Test 12: Full pipeline with checkpoint (if available)."""
    runner.log("Test: Full pipeline with checkpoint")
    
    checkpoint_path = 'c:/Users/mlaitenberger/OneDrive - Deloitte (O365D)/Desktop/Scenario_Project/scenario_generation/results/germany_boom_24/checkpoint-10.pt'
    
    if not os.path.exists(checkpoint_path):
        runner.log(f"  Checkpoint not found, skipping: {checkpoint_path}", 'WARN')
        return
    
    config = {
        'data_hist_path': 'scenario_generation/data/germany_macro_augemented.csv',
        'data_stress_path': 'scenario_generation/data/germany_stress_boom.csv',
        'seq_length': 24,
        'feature_size': 5,
        'num_samples': 5,
        'sampling_steps': 50,
        'results_folder': os.path.join(runner.base_results_folder, 'test12'),
        'batch_size': 16,
        'milestone': checkpoint_path,
        'stressed_features': [0, 1],
        'stressed_seq_indices': [6, 7, 8, 9, 10],
        'len_historie': 6,
    }
    
    handler = Handler('diffusion_ts', config)
    runner.log(f"  Handler with checkpoint initialized successfully")


def main():
    runner = TestRunner()
    
    runner.log("=" * 60)
    runner.log("SCENARIO TESTING SUITE")
    runner.log("=" * 60)
    
    # Data processing tests
    runner.run_test("Test 1: InputData loading", lambda: test_input_data_loading(runner))
    runner.run_test("Test 2: Overlapping sequences", lambda: test_overlapping_sequences(runner))
    runner.run_test("Test 3: Forecast with stress", lambda: test_forecast_seq_with_stress(runner))
    runner.run_test("Test 4: Forecast without stress", lambda: test_forecast_seq_without_stress(runner))
    runner.run_test("Test 5: Forecast with None len_historie", lambda: test_forecast_seq_with_none_len_historie(runner))
    
    # Handler tests
    runner.run_test("Test 6: Handler basic config", lambda: test_handler_basic_config(runner))
    runner.run_test("Test 7: Handler with stress", lambda: test_handler_with_stress(runner))
    
    # Edge cases
    runner.run_test("Test 8: Different sequence lengths", lambda: test_different_seq_lengths(runner))
    runner.run_test("Test 9: Single stress index", lambda: test_edge_case_single_stress_index(runner))
    runner.run_test("Test 10: Config validation", lambda: test_config_validation(runner))
    runner.run_test("Test 11: Data shape consistency", lambda: test_data_shape_consistency(runner))
    runner.run_test("Test 12: Full pipeline with checkpoint", lambda: test_full_pipeline_with_checkpoint(runner))
    
    runner.print_summary()
    
    return 0 if all(r['status'] == 'PASSED' for r in runner.results) else 1


if __name__ == '__main__':
    sys.exit(main())
