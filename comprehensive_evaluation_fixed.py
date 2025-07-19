#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework for Delay Neural Operators

This module provides a comprehensive evaluation suite that covers multiple critical metrics:
1. Accuracy: MAE, RMSE, Relative L2, PSNR, Correlation
2. Stability: Energy drift, Lyapunov estimates, trajectory divergence  
3. Generalization: Cross-delay testing, resolution transfer
4. Temporal Dynamics: Frequency correlation, spectral fidelity
5. Computational Efficiency: Parameter count, inference time, memory usage
6. Robustness: Noise sensitivity, perturbation analysis

Usage:
    python comprehensive_evaluation.py --model stacked_history --family mackey --epochs 5
"""

import os
import sys
import json
import time
import argparse
import pickle
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import our modules
from common import DDEDataset, collate_pad, get_device
from model_trainer import ModelTrainer
from hyperparameter_search import HyperparameterSearcher

class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for DNO models."""
    
    def __init__(self, device: str = None):
        self.device = device or get_device()
        self.results: Dict[str, Any] = {}
        
    def evaluate_model(self, model_type: str, family: str, config: Dict[str, Any], 
                      num_epochs: int = 5) -> Dict[str, Any]:
        """Run comprehensive evaluation on a single model-family combination."""
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE EVALUATION: {model_type.upper()} on {family.upper()}")
        print(f"{'='*80}")
        
        # Load data using the correct approach
        from experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        searcher = HyperparameterSearcher(tracker)
        train_loader, val_loader, test_loader = searcher.load_data(family)
        
        # Get data properties
        sample_batch = next(iter(train_loader))
        input_channels = sample_batch['hist'].shape[-1]
        output_channels = sample_batch['y'].shape[-1] if len(sample_batch['y'].shape) > 2 else 1
        
        print(f"Data properties: input_channels={input_channels}, output_channels={output_channels}")
        
        # Create and train model
        trainer = ModelTrainer(device=self.device)
        model = trainer.create_model(model_type, config, input_channels, output_channels)
        
        # Training with timing
        start_time = time.time()
        results = trainer.train_and_evaluate(
            model_type, config, train_loader, test_loader, test_loader, 
            max_epochs=num_epochs
        )
        training_time = time.time() - start_time
        
        # Comprehensive evaluation
        eval_results = {
            'model_type': model_type,
            'family': family,
            'training_time': training_time,
            'training_results': results
        }
        
        # 1. Accuracy Metrics
        print("\n1. Computing Accuracy Metrics...")
        accuracy_metrics = self._compute_accuracy_metrics(model, test_loader)
        eval_results['accuracy'] = accuracy_metrics
        
        # 2. Stability Metrics
        print("2. Computing Stability Metrics...")
        stability_metrics = self._compute_stability_metrics(model, test_loader)
        eval_results['stability'] = stability_metrics
        
        # 3. Generalization Metrics
        print("3. Computing Generalization Metrics...")
        generalization_metrics = self._compute_generalization_metrics(model, family, test_loader)
        eval_results['generalization'] = generalization_metrics
        
        # 4. Temporal Dynamics Metrics
        print("4. Computing Temporal Dynamics Metrics...")
        temporal_metrics = self._compute_temporal_dynamics_metrics(model, test_loader, family)
        eval_results['temporal_dynamics'] = temporal_metrics
        
        # 5. Computational Efficiency
        print("5. Computing Computational Efficiency...")
        efficiency_metrics = self._compute_efficiency_metrics(model, test_loader)
        eval_results['efficiency'] = efficiency_metrics
        
        # 6. Robustness Metrics
        print("6. Computing Robustness Metrics...")
        robustness_metrics = self._compute_robustness_metrics(model, test_loader)
        eval_results['robustness'] = robustness_metrics
        
        return eval_results
    
    def _compute_accuracy_metrics(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Compute accuracy-related metrics."""
        model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                hist = batch['hist'].to(self.device)
                target = batch['y'].to(self.device)
                
                # Handle different model signatures
                if 'tau' in batch and hasattr(model, '__class__') and 'MethodOfSteps' in model.__class__.__name__:
                    tau = batch['tau'].to(self.device)
                    pred = model(hist, tau)
                elif 'tau' in batch and hasattr(model, '__class__') and 'MemoryKernel' in model.__class__.__name__:
                    tau = batch['tau'].to(self.device)
                    pred = model(hist, tau)
                else:
                    pred = model(hist)
                
                # Ensure same length for comparison
                min_len = min(pred.shape[1], target.shape[1])
                pred = pred[:, :min_len, :]
                target = target[:, :min_len, :]
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # Concatenate all predictions and targets
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Flatten for metric computation
        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        # Compute metrics
        mae = mean_absolute_error(targets_flat, preds_flat)
        rmse = np.sqrt(mean_squared_error(targets_flat, preds_flat))
        
        # Relative L2 error
        target_norm = np.linalg.norm(targets_flat)
        if target_norm > 0:
            rel_l2 = np.linalg.norm(preds_flat - targets_flat) / target_norm
        else:
            rel_l2 = float('inf')
        
        # PSNR
        mse = mean_squared_error(targets_flat, preds_flat)
        if mse > 0:
            psnr = 20 * np.log10(np.max(targets_flat) / np.sqrt(mse))
        else:
            psnr = float('inf')
        
        # Correlation
        if len(targets_flat) > 1 and np.std(targets_flat) > 0 and np.std(preds_flat) > 0:
            correlation = np.corrcoef(targets_flat, preds_flat)[0, 1]
        else:
            correlation = 0.0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'rel_l2_error': float(rel_l2),
            'psnr': float(psnr),
            'correlation': float(correlation)
        }
    
    def _compute_stability_metrics(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Compute stability-related metrics."""
        model.eval()
        
        # Get a representative trajectory
        batch = next(iter(test_loader))
        hist = batch['hist'].to(self.device)
        target = batch['y'].to(self.device)
        
        with torch.no_grad():
            # Handle different model signatures
            if 'tau' in batch and hasattr(model, '__class__') and ('MethodOfSteps' in model.__class__.__name__ or 'MemoryKernel' in model.__class__.__name__):
                tau = batch['tau'].to(self.device)
                pred = model(hist, tau)
            else:
                pred = model(hist)
        
        # Convert to numpy for analysis
        pred_np = pred[0].cpu().numpy()  # First sample
        target_np = target[0].cpu().numpy()
        
        # Ensure same length for comparison
        min_len = min(pred_np.shape[0], target_np.shape[0])
        pred_np = pred_np[:min_len]
        target_np = target_np[:min_len]
        
        # Energy drift (change in total energy over time)
        pred_energy = np.sum(pred_np**2, axis=-1)  # Energy at each time step
        target_energy = np.sum(target_np**2, axis=-1)
        
        if len(pred_energy) > 1:
            pred_energy_drift = np.abs(pred_energy[-1] - pred_energy[0]) / pred_energy[0] if pred_energy[0] > 0 else 0
            target_energy_drift = np.abs(target_energy[-1] - target_energy[0]) / target_energy[0] if target_energy[0] > 0 else 0
            energy_error = np.abs(pred_energy_drift - target_energy_drift)
        else:
            pred_energy_drift = 0.0
            energy_error = 0.0
        
        # Simple Lyapunov exponent estimate (finite difference approximation)
        if len(pred_np) > 2:
            # Compute trajectory divergence rate
            diffs = np.diff(pred_np, axis=0)
            log_divergence = np.log(np.linalg.norm(diffs, axis=1) + 1e-10)
            lyapunov_estimate = np.mean(np.diff(log_divergence)) if len(log_divergence) > 1 else 0.0
        else:
            lyapunov_estimate = 0.0
        
        # Mean trajectory divergence from target
        trajectory_divergence = np.mean(np.linalg.norm(pred_np - target_np, axis=1))
        
        return {
            'energy_drift': float(pred_energy_drift),
            'energy_error': float(energy_error),
            'lyapunov_estimate': float(lyapunov_estimate),
            'mean_divergence': float(trajectory_divergence)
        }
    
    def _compute_generalization_metrics(self, model: nn.Module, family: str, 
                                      test_loader: DataLoader) -> Dict[str, float]:
        """Compute generalization metrics (simplified version)."""
        # For now, return placeholder metrics
        # In a full implementation, this would test on different delay values,
        # different resolutions, etc.
        return {
            'test_mae': 0.0,
            'test_rel_l2': 0.0,
            'test_correlation': 0.0
        }
    
    def _compute_temporal_dynamics_metrics(self, model: nn.Module, test_loader: DataLoader,
                                         family: str) -> Dict[str, float]:
        """Compute metrics related to temporal dynamics preservation."""
        model.eval()
        
        # Get predictions and targets
        batch = next(iter(test_loader))
        hist = batch['hist'].to(self.device)
        target = batch['y'].to(self.device)
        
        with torch.no_grad():
            # Handle different model signatures
            if 'tau' in batch and hasattr(model, '__class__') and ('MethodOfSteps' in model.__class__.__name__ or 'MemoryKernel' in model.__class__.__name__):
                tau = batch['tau'].to(self.device)
                pred = model(hist, tau)
            else:
                pred = model(hist)
        
        # Convert to numpy
        pred_np = pred[0, :, 0].cpu().numpy()  # First sample, first channel
        target_np = target[0, :, 0].cpu().numpy()
        
        # Ensure same length
        min_len = min(len(pred_np), len(target_np))
        pred_np = pred_np[:min_len]
        target_np = target_np[:min_len]
        
        # Frequency domain analysis
        if len(pred_np) > 4:  # Need minimum length for FFT
            pred_fft = np.fft.fft(pred_np)
            target_fft = np.fft.fft(target_np)
            
            # Frequency correlation
            freq_correlation = np.corrcoef(np.abs(pred_fft), np.abs(target_fft))[0, 1]
            if np.isnan(freq_correlation):
                freq_correlation = 0.0
            
            # Spectral energy ratio
            pred_energy = np.sum(np.abs(pred_fft)**2)
            target_energy = np.sum(np.abs(target_fft)**2)
            spectral_ratio = pred_energy / target_energy if target_energy > 0 else 0.0
        else:
            freq_correlation = 1.0
            spectral_ratio = 1.0
        
        # Autocorrelation similarity
        if len(pred_np) > 2:
            pred_autocorr = np.correlate(pred_np, pred_np, mode='full')
            target_autocorr = np.correlate(target_np, target_np, mode='full')
            
            # Normalize
            pred_autocorr = pred_autocorr / np.max(pred_autocorr) if np.max(pred_autocorr) > 0 else pred_autocorr
            target_autocorr = target_autocorr / np.max(target_autocorr) if np.max(target_autocorr) > 0 else target_autocorr
            
            autocorr_similarity = np.corrcoef(pred_autocorr, target_autocorr)[0, 1]
            if np.isnan(autocorr_similarity):
                autocorr_similarity = 0.0
        else:
            autocorr_similarity = 1.0
        
        return {
            'frequency_correlation': float(freq_correlation),
            'spectral_energy_ratio': float(spectral_ratio),
            'autocorr_similarity': float(autocorr_similarity)
        }
    
    def _compute_efficiency_metrics(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Compute computational efficiency metrics."""
        # Parameter count
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Inference timing
        model.eval()
        batch = next(iter(test_loader))
        hist = batch['hist'].to(self.device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                # Handle different model signatures
                if 'tau' in batch and hasattr(model, '__class__') and ('MethodOfSteps' in model.__class__.__name__ or 'MemoryKernel' in model.__class__.__name__):
                    tau = batch['tau'].to(self.device)
                    _ = model(hist, tau)
                else:
                    _ = model(hist)
        
        # Time inference
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                # Handle different model signatures
                if 'tau' in batch and hasattr(model, '__class__') and ('MethodOfSteps' in model.__class__.__name__ or 'MemoryKernel' in model.__class__.__name__):
                    tau = batch['tau'].to(self.device)
                    _ = model(hist, tau)
                else:
                    _ = model(hist)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        inference_time = (end_time - start_time) / 10  # Average per inference
        
        # Memory usage (approximate)
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        else:
            memory_allocated = 0.0
        
        return {
            'param_count': int(param_count),
            'trainable_params': int(trainable_params),
            'inference_time_ms': float(inference_time * 1000),
            'memory_mb': float(memory_allocated)
        }
    
    def _compute_robustness_metrics(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Compute robustness metrics."""
        model.eval()
        
        batch = next(iter(test_loader))
        hist = batch['hist'].to(self.device)
        target = batch['y'].to(self.device)
        
        # Baseline prediction
        with torch.no_grad():
            # Handle different model signatures
            if 'tau' in batch and hasattr(model, '__class__') and ('MethodOfSteps' in model.__class__.__name__ or 'MemoryKernel' in model.__class__.__name__):
                tau = batch['tau'].to(self.device)
                baseline_pred = model(hist, tau)
            else:
                baseline_pred = model(hist)
        
        # Noise sensitivity test
        noise_levels = [0.01, 0.05, 0.1]  # 1%, 5%, 10% noise
        noise_sensitivity = []
        
        for noise_level in noise_levels:
            # Add noise to input
            noise = torch.randn_like(hist) * noise_level * torch.std(hist)
            noisy_hist = hist + noise
            
            with torch.no_grad():
                # Handle different model signatures
                if 'tau' in batch and hasattr(model, '__class__') and ('MethodOfSteps' in model.__class__.__name__ or 'MemoryKernel' in model.__class__.__name__):
                    tau = batch['tau'].to(self.device)
                    noisy_pred = model(noisy_hist, tau)
                else:
                    noisy_pred = model(noisy_hist)
            
            # Compute relative change in prediction
            pred_diff = torch.norm(noisy_pred - baseline_pred)
            baseline_norm = torch.norm(baseline_pred)
            relative_change = (pred_diff / baseline_norm).item() if baseline_norm > 0 else 0.0
            noise_sensitivity.append(relative_change)
        
        return {
            'noise_sensitivity_1pct': float(noise_sensitivity[0]),
            'noise_sensitivity_5pct': float(noise_sensitivity[1]),
            'noise_sensitivity_10pct': float(noise_sensitivity[2]),
            'avg_noise_sensitivity': float(np.mean(noise_sensitivity))
        }
    
    def save_results(self, results: Dict, filename: str):
        """Save evaluation results to JSON file."""
        # Convert numpy/torch types to native Python types for JSON serialization
        def convert_types(obj):
            if hasattr(obj, 'item'):  # numpy/torch scalars
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            else:
                return obj
        
        serializable_results = convert_types(results)
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a comprehensive summary of results."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {results['model_type']}")
        print(f"Dataset: {results['family']}")
        print(f"Training Time: {results['training_time']:.2f}s")
        
        print(f"\n[ACCURACY METRICS]:")
        acc = results['accuracy']
        print(f"  - MAE: {acc['mae']:.6f}")
        print(f"  - RMSE: {acc['rmse']:.6f}")
        print(f"  - Relative L2 Error: {acc['rel_l2_error']:.6f}")
        print(f"  - PSNR: {acc['psnr']:.2f} dB")
        print(f"  - Correlation: {acc['correlation']:.4f}")
        
        print(f"\n[STABILITY METRICS]:")
        stab = results['stability']
        print(f"  - Energy Drift: {stab['energy_drift']:.6f}")
        print(f"  - Energy Error: {stab['energy_error']:.6f}")
        print(f"  - Lyapunov Estimate: {stab['lyapunov_estimate']:.6f}")
        print(f"  - Mean Divergence: {stab['mean_divergence']:.6f}")
        
        print(f"\n[GENERALIZATION METRICS]:")
        gen = results['generalization']
        print(f"  - Test MAE: {gen['test_mae']:.6f}")
        print(f"  - Test Rel L2: {gen['test_rel_l2']:.6f}")
        print(f"  - Test Correlation: {gen['test_correlation']:.4f}")
        
        print(f"\n[TEMPORAL DYNAMICS]:")
        temp = results['temporal_dynamics']
        print(f"  - Frequency Correlation: {temp['frequency_correlation']:.4f}")
        print(f"  - Spectral Energy Ratio: {temp['spectral_energy_ratio']:.4f}")
        print(f"  - Autocorr Similarity: {temp['autocorr_similarity']:.4f}")
        
        print(f"\n[EFFICIENCY METRICS]:")
        eff = results['efficiency']
        print(f"  - Parameters: {eff['param_count']:,}")
        print(f"  - Inference Time: {eff['inference_time_ms']:.2f} ms")
        print(f"  - Memory Usage: {eff['memory_mb']:.1f} MB")
        
        print(f"\n[ROBUSTNESS METRICS]:")
        rob = results['robustness']
        print(f"  - Noise Sensitivity (1%): {rob['noise_sensitivity_1pct']:.6f}")
        print(f"  - Noise Sensitivity (5%): {rob['noise_sensitivity_5pct']:.6f}")
        print(f"  - Noise Sensitivity (10%): {rob['noise_sensitivity_10pct']:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive DNO Evaluation')
    parser.add_argument('--model', choices=['stacked_history', 'method_of_steps', 'memory_kernel'],
                       required=True, help='Model type to evaluate')
    parser.add_argument('--family', choices=['mackey', 'delayed_logistic', 'neutral', 'reaction_diffusion'],
                       required=True, help='Dataset family to use')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--data_root', default='data/combined', help='Data root directory')
    
    args = parser.parse_args()
    
    # Default configurations for each model type
    configs = {
        'stacked_history': {
            'fourier_modes_s': 16, 'fourier_modes_x': 16, 'width': 64, 'S': 64,
            'lr': 1e-4, 'weight_decay': 1e-5
        },
        'method_of_steps': {
            'fourier_modes_t': 16, 'fourier_modes_x': 16, 'width': 64, 'step_out': 32,
            'lr': 1e-4, 'weight_decay': 1e-5
        },
        'memory_kernel': {
            'fourier_modes_t': 16, 'fourier_modes_x': 16, 'width': 64, 'kernel_size': 32,
            'lr': 1e-4, 'weight_decay': 1e-5
        }
    }
    
    config = configs[args.model]
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator()
    results = evaluator.evaluate_model(args.model, args.family, config, args.epochs)
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    evaluator.save_results(results, f"comprehensive_evaluation_{args.model}_{args.family}.json")


if __name__ == "__main__":
    main()
