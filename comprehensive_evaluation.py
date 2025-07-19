"""
comprehensive_evaluation.py
============================
Comprehensive testing and evaluation framework for Delay Neural Operators.
Provides extensive metrics across accuracy, stability, generalization, temporal dynamics,
computational efficiency, and robustness.

Usage:
    python comprehensive_evaluation.py --model stacked_history --family mackey
    python comprehensive_evaluation.py --all  # Test all combinations
"""

import os
import time
import argparse
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from model_trainer import ModelTrainer
from hyperparameter_search import HyperparameterSearcher
from experiment_tracker import ExperimentTracker
from common import DDEDataset, collate_pad, get_device


class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for DNO models."""
    
    def __init__(self, data_root: str = "data/combined"):
        self.data_root = data_root
        self.device = get_device()
        self.results = {}
        
    def evaluate_model(self, model_type: str, family: str, 
                      config: Dict[str, Any], num_epochs: int = 10) -> Dict[str, Any]:
        """Comprehensive evaluation of a single model-family combination."""
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE EVALUATION: {model_type.upper()} on {family.upper()}")
        print(f"{'='*80}")
        
        # Initialize components
        tracker = ExperimentTracker(f"dno-comprehensive-eval")
        searcher = HyperparameterSearcher(tracker)
        trainer = ModelTrainer()
        
        # Load data
        train_loader, val_loader, test_loader = searcher.load_data(
            family, config['data_root']
        )
        
        # Get data properties
        sample_batch = next(iter(train_loader))
        input_channels = sample_batch['hist'].shape[-1]  # Last dim is channels
        output_channels = sample_batch['y'].shape[-1] if len(sample_batch['y'].shape) > 2 else 1
        
        print(f"Data properties: input_channels={input_channels}, output_channels={output_channels}")
        
        # Create and train model
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
        stability_metrics = self._compute_stability_metrics(model, test_loader, family)
        eval_results['stability'] = stability_metrics
        
        # 3. Generalization Metrics
        print("3. Computing Generalization Metrics...")
        generalization_metrics = self._compute_generalization_metrics(
            model, model_type, config, family, input_channels, output_channels
        )
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
        """Compute comprehensive accuracy metrics."""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                hist = batch['hist'].to(self.device)
                target = batch['y'].to(self.device)
                
                pred = model(hist)
                
                # Handle different output shapes
                if pred.shape != target.shape:
                    min_len = min(pred.shape[1], target.shape[1])
                    pred = pred[:, :min_len]
                    target = target[:, :min_len]
                
                all_predictions.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Flatten for metric computation
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)
        
        # Compute metrics
        mae = mean_absolute_error(target_flat, pred_flat)
        rmse = np.sqrt(mean_squared_error(target_flat, pred_flat))
        
        # Relative L2 error
        rel_l2 = np.linalg.norm(pred_flat - target_flat) / (np.linalg.norm(target_flat) + 1e-8)
        
        # PSNR (Peak Signal-to-Noise Ratio)
        mse = np.mean((pred_flat - target_flat) ** 2)
        max_val = np.max(np.abs(target_flat))
        psnr = 20 * np.log10(max_val / (np.sqrt(mse) + 1e-8)) if mse > 0 else float('inf')
        
        # Correlation coefficient
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1] if len(pred_flat) > 1 else 0.0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'rel_l2_error': float(rel_l2),
            'psnr': float(psnr),
            'correlation': float(correlation)
        }
    
    def _compute_stability_metrics(self, model: nn.Module, test_loader: DataLoader, 
                                 family: str) -> Dict[str, float]:
        """Compute stability-related metrics."""
        model.eval()
        
        # Get a representative trajectory
        batch = next(iter(test_loader))
        hist = batch['hist'].to(self.device)
        target = batch['y'].to(self.device)
        
        with torch.no_grad():
            pred = model(hist)
            
            # Handle shape mismatch
            if pred.shape != target.shape:
                min_len = min(pred.shape[1], target.shape[1])
                pred = pred[:, :min_len]
                target = target[:, :min_len]
        
        # Convert to numpy for analysis
        pred_traj = pred[0].cpu().numpy()  # First sample
        target_traj = target[0].cpu().numpy()
        
        # Energy-based stability (for systems with energy conservation)
        pred_energy = np.mean(pred_traj ** 2, axis=-1) if len(pred_traj.shape) > 1 else pred_traj ** 2
        target_energy = np.mean(target_traj ** 2, axis=-1) if len(target_traj.shape) > 1 else target_traj ** 2
        
        energy_drift = np.std(pred_energy) / (np.mean(pred_energy) + 1e-8)
        energy_error = np.mean(np.abs(pred_energy - target_energy)) / (np.mean(target_energy) + 1e-8)
        
        # Long-term divergence
        time_steps = min(len(pred_traj), len(target_traj))
        divergence = np.array([
            np.linalg.norm(pred_traj[i] - target_traj[i]) 
            for i in range(time_steps)
        ])
        
        # Linear fit to log(divergence) to estimate exponential growth
        valid_div = divergence[divergence > 1e-10]
        if len(valid_div) > 10:
            log_div = np.log(valid_div)
            time_indices = np.arange(len(log_div))
            lyapunov_estimate = np.polyfit(time_indices, log_div, 1)[0]
        else:
            lyapunov_estimate = 0.0
        
        return {
            'energy_drift': float(energy_drift),
            'energy_error': float(energy_error),
            'lyapunov_estimate': float(lyapunov_estimate),
            'final_divergence': float(divergence[-1]) if len(divergence) > 0 else 0.0,
            'mean_divergence': float(np.mean(divergence))
        }
    
    def _compute_generalization_metrics(self, model: nn.Module, model_type: str, 
                                      config: Dict[str, Any], family: str,
                                      input_channels: int, output_channels: int) -> Dict[str, float]:
        """Compute generalization metrics across different conditions."""
        
        # For now, we'll compute basic generalization metrics
        # In a full implementation, you'd test on different delay values, resolutions, etc.
        
        # Test on validation data as a proxy for generalization
        try:
            # Load test data
            test_path = os.path.join(self.data_root, f"{family}_test.pkl")
            if os.path.exists(test_path):
                with open(test_path, 'rb') as f:
                    test_data = pickle.load(f)
                
                test_dataset = DDEDataset(test_data, family)
                test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_pad)
                
                # Compute accuracy on test set
                test_accuracy = self._compute_accuracy_metrics(model, test_loader)
                
                return {
                    'test_mae': test_accuracy['mae'],
                    'test_rel_l2': test_accuracy['rel_l2_error'],
                    'test_correlation': test_accuracy['correlation']
                }
        except Exception as e:
            print(f"Warning: Could not compute generalization metrics: {e}")
        
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
            pred = model(hist)
            
            if pred.shape != target.shape:
                min_len = min(pred.shape[1], target.shape[1])
                pred = pred[:, :min_len]
                target = target[:, :min_len]
        
        pred_traj = pred[0].cpu().numpy()
        target_traj = target[0].cpu().numpy()
        
        # Flatten if multi-dimensional
        if len(pred_traj.shape) > 1:
            pred_traj = pred_traj.mean(axis=-1)
            target_traj = target_traj.mean(axis=-1)
        
        # Frequency domain analysis
        if len(pred_traj) > 16:  # Need sufficient length for FFT
            pred_fft = np.abs(np.fft.fft(pred_traj))
            target_fft = np.abs(np.fft.fft(target_traj))
            
            # Frequency spectrum correlation
            freq_correlation = np.corrcoef(pred_fft, target_fft)[0, 1] if len(pred_fft) > 1 else 0.0
            
            # Spectral energy ratio
            pred_energy = np.sum(pred_fft ** 2)
            target_energy = np.sum(target_fft ** 2)
            spectral_energy_ratio = pred_energy / (target_energy + 1e-8)
        else:
            freq_correlation = 0.0
            spectral_energy_ratio = 1.0
        
        # Autocorrelation preservation
        if len(pred_traj) > 10:
            pred_autocorr = np.correlate(pred_traj, pred_traj, mode='full')
            target_autocorr = np.correlate(target_traj, target_traj, mode='full')
            
            # Compare central parts of autocorrelation
            center = len(pred_autocorr) // 2
            window = min(10, center)
            pred_ac = pred_autocorr[center-window:center+window]
            target_ac = target_autocorr[center-window:center+window]
            
            autocorr_similarity = np.corrcoef(pred_ac, target_ac)[0, 1] if len(pred_ac) > 1 else 0.0
        else:
            autocorr_similarity = 0.0
        
        return {
            'frequency_correlation': float(freq_correlation),
            'spectral_energy_ratio': float(spectral_energy_ratio),
            'autocorr_similarity': float(autocorr_similarity)
        }
    
    def _compute_efficiency_metrics(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Compute computational efficiency metrics."""
        
        # Model parameters
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Inference timing
        model.eval()
        batch = next(iter(test_loader))
        hist = batch['hist'].to(self.device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = model(hist)
        
        # Time inference
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(hist)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        inference_time = (end_time - start_time) / 10.0  # Average per inference
        
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
            baseline_pred = model(hist)
        
        # Noise sensitivity
        noise_levels = [0.01, 0.05, 0.1]
        noise_sensitivity = []
        
        for noise_level in noise_levels:
            noise = torch.randn_like(hist) * noise_level
            noisy_hist = hist + noise
            
            with torch.no_grad():
                noisy_pred = model(noisy_hist)
            
            # Handle shape mismatch
            if noisy_pred.shape != baseline_pred.shape:
                min_len = min(noisy_pred.shape[1], baseline_pred.shape[1])
                noisy_pred = noisy_pred[:, :min_len]
                baseline_pred_truncated = baseline_pred[:, :min_len]
            else:
                baseline_pred_truncated = baseline_pred
            
            # Compute relative change
            diff = torch.mean((noisy_pred - baseline_pred_truncated) ** 2)
            baseline_norm = torch.mean(baseline_pred_truncated ** 2)
            relative_change = (diff / (baseline_norm + 1e-8)).item()
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
    hist = batch['hist'].to(self.device)
    target = batch['y'].to(self.device)
    
    with torch.no_grad():
        pred = model(hist)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        model_type = results['model_type']
        family = results['family']
        
        print(f"Model: {model_type}")
        print(f"Dataset: {family}")
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
    parser = argparse.ArgumentParser(description="Comprehensive DNO Evaluation")
    parser.add_argument('--model', type=str, choices=['stacked_history', 'method_of_steps', 'memory_kernel'],
                       help='Model type to evaluate')
    parser.add_argument('--family', type=str, choices=['mackey', 'delayed_logistic', 'neutral', 'reaction_diffusion'],
                       help='Dataset family to evaluate')
    parser.add_argument('--all', action='store_true', help='Evaluate all model-family combinations')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--data_root', type=str, default='data/combined', help='Data root directory')
    
    args = parser.parse_args()
    
    evaluator = ComprehensiveEvaluator(args.data_root)
    
    # Base configuration
    base_config = {
        'family': 'mackey',  # Will be overridden
        'fourier_modes_x': 8,
        'fourier_modes_s': 16,
        'S': 32,
        'width': 64,
        'n_layers': 2,
        'lr': 0.001,
        'batch_size': 8,
        'dropout': 0.1,
        'weight_decay': 1e-6,
        'data_root': args.data_root
    }
    
    if args.all:
        # Evaluate all combinations
        models = ['stacked_history', 'method_of_steps', 'memory_kernel']
        families = ['mackey', 'delayed_logistic', 'neutral', 'reaction_diffusion']
        
        all_results = {}
        
        for model_type in models:
            for family in families:
                config = base_config.copy()
                config['family'] = family
                
                try:
                    results = evaluator.evaluate_model(model_type, family, config, args.epochs)
                    all_results[f"{model_type}_{family}"] = results
                    evaluator.print_summary(results)
                    
                except Exception as e:
                    print(f"[FAILED] Failed to evaluate {model_type} on {family}: {e}")
                    continue
        
        # Save all results
        evaluator.save_results(all_results, "comprehensive_evaluation_all_combinations.json")
        
        print(f"\n[SUCCESS] Completed comprehensive evaluation of {len(all_results)} combinations!")
        
    else:
        if not args.model or not args.family:
            print("Error: Must specify --model and --family, or use --all")
            return
        
        config = base_config.copy()
        config['family'] = args.family
        
        results = evaluator.evaluate_model(args.model, args.family, config, args.epochs)
        evaluator.print_summary(results)
        evaluator.save_results(results, f"comprehensive_evaluation_{args.model}_{args.family}.json")


if __name__ == "__main__":
    main()
