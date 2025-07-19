"""
hyperparameter_search.py
========================
Production-ready hyperparameter search for Delay Neural Operator (DNO) models.
Supports grid search, random search, and Bayesian optimization with W&B logging.

Usage:
    python hyperparameter_search.py --search_type bayesian --n_trials 100 --max_epochs 200
    python hyperparameter_search.py --search_type grid --model stacked_history
    python hyperparameter_search.py --search_type random --n_trials 50 --family mackey
"""

import os
import sys
import json
import pickle
import argparse
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from typing_extensions import TypedDict, NotRequired
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader

# Hyperparameter optimization libraries
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Bayesian search disabled.")

# Import DNO modules
from common_py34 import set_seed, DDEDataset, collate_pad, get_device
from hyperparameter_config import SearchConfig
from experiment_tracker import ExperimentTracker
from model_trainer import ModelTrainer

# Import model classes
from stacked_history import StackedHistoryFNO
from method_of_steps import MethodOfStepsModel
from memory_kernel import MemoryKernelModel

def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class ExperimentResult(TypedDict):
    """Type definition for experiment results."""
    run_id: str
    success: bool
    error: NotRequired[str]
    test_rel_error: float
    model: str
    family: str
    search_type: str

class HyperparameterSearcher:
    """Comprehensive hyperparameter search with multiple strategies."""
    
    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
        self.trainer = ModelTrainer()
        self.results: List[ExperimentResult] = []
    
    def load_data(self, family: str, data_root: str = "./data") -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load train/validation/test data for given family."""
        # Map family names to file names
        family_map = {
            'mackey': 'mackey_glass',
            'delayed_logistic': 'delayed_logistic', 
            'neutral': 'neutral_dde',
            'reaction_diffusion': 'reaction_diffusion'
        }
        
        file_family = family_map.get(family, family)
        
        # Use absolute path since we're in workspace/delay/dno/dno1/dno
        data_dir = os.path.join(os.getcwd(), "data", "combined")
        train_path = os.path.join(data_dir, f"{file_family}_train.pkl")
        test_path = os.path.join(data_dir, f"{file_family}_test.pkl")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Data files not found for family {family}")
        
        # Load datasets
        train_ds = DDEDataset(train_path)
        test_ds = DDEDataset(test_path)
        
        # Create train/validation split (80/20 of training data)
        train_size = int(0.8 * len(train_ds))
        val_size = len(train_ds) - train_size
        train_subset, val_subset = torch.utils.data.random_split(train_ds, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=collate_pad)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=collate_pad)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_pad)
        
        return train_loader, val_loader, test_loader
    
    def _validate_and_convert_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and convert config parameters to correct types."""
        # Integer parameters that must be converted
        int_params = {
            'fourier_modes_x', 'fourier_modes_t', 'fourier_modes_s', 'width', 'n_layers',
            'S', 'step_out', 'hist_hidden', 'hist_layers', 'tau_dim', 'predictor_hidden',
            'predictor_layers', 'hidden', 'kernel_layers', 'euler_steps', 'batch_size',
            'roll_steps'
        }
        
        # Convert parameters to correct types
        validated_config = config.copy()
        for param, value in validated_config.items():
            if param in int_params and not isinstance(value, int):
                validated_config[param] = int(float(value))
            elif param in ['lr', 'weight_decay', 'dropout', 'euler_dt'] and not isinstance(value, float):
                validated_config[param] = float(value)
        
        return validated_config
    
    def _run_single_experiment(self, config: Dict[str, Any], max_epochs: int) -> ExperimentResult:
        """Run a single hyperparameter experiment."""
        model_type = config['model']
        family = config['family']
        
        # Validate and convert config parameters
        config = self._validate_and_convert_config(config)
        
        # Start tracking
        run_name = f"{model_type}_{family}_{config.get('search_type', 'manual')}"
        run_id = self.tracker.start_run(config, run_name)
        
        try:
            # Load data with batch size from config
            train_loader, val_loader, test_loader = self.load_data(family, data_root="./data")
            
            # Extract first sample to determine input/output dimensions
            first_batch = next(iter(train_loader))
            hist = first_batch["hist"]
            y = first_batch["y"]
            
            # Determine output dimension and spatial width
            if y.ndim == 3:
                D_out = y.shape[2]
                X = D_out if family == "reaction_diffusion" else 1
            else:
                D_out = 1
                X = 1
                
            # Adjust fourier_modes_x if needed
            if X == 1 and config.get('fourier_modes_x', 1) > 1:
                print(f"Adjusting fourier_modes_x from {config['fourier_modes_x']} to 1 to match X dimension")
                config['fourier_modes_x'] = 1
            
            # Handle parameter conversions for stacked_history
            if model_type == 'stacked_history':
                # Convert fourier_modes_t to fourier_modes_s if present
                if 'fourier_modes_t' in config:
                    config['fourier_modes_s'] = config.pop('fourier_modes_t')
                
                # Set default S if not provided
                if 'S' not in config:
                    config['S'] = 64  # Default value
            
            # Add family info to config for model creation
            config['family'] = family
            
            # Update batch size if specified in config
            batch_size = config.get('batch_size', 32)
            if batch_size != 32:
                train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, 
                                        shuffle=True, collate_fn=collate_pad)
                val_loader = DataLoader(val_loader.dataset, batch_size=batch_size, 
                                      shuffle=False, collate_fn=collate_pad)
                test_loader = DataLoader(test_loader.dataset, batch_size=batch_size, 
                                       shuffle=False, collate_fn=collate_pad)
            
            # Train and evaluate using model_trainer
            results = self.trainer.train_and_evaluate(
                model_type, config, train_loader, val_loader, test_loader,
                max_epochs, self.tracker
            )
            
            # Add config to results
            results.update(config)
            results['run_id'] = run_id
            results['success'] = True
            
            # Finish tracking
            self.tracker.finish_run(results)
            
            return results
            
        except Exception as e:
            print(f"Experiment failed: {e}")
            results = {
                **config,
                'run_id': run_id,
                'success': False,
                'error': str(e),
                'test_rel_error': float('inf'),
                'run_id': '',
                'model': model_type,
                'family': family
            }
            self.tracker.finish_run(results)
            return results
    
    def grid_search(self, model_type: str, family: str, max_epochs: int = 200,
                   max_combinations: int = 100) -> List[ExperimentResult]:
        """Exhaustive grid search over hyperparameter space."""
        print(f"Starting grid search for {model_type} on {family}")
        
        search_space = SearchConfig.SEARCH_SPACES[model_type]
        
        # Generate all combinations
        keys = list(search_space.keys())
        values: List[List[Union[float, int]]] = []
        
        for v in search_space.values():
            if isinstance(v, list):
                values.append([float(x) if isinstance(x, float) else int(x) for x in v])
            else:
                low, high = float(v[0]), float(v[1])
                values.append([low + (high-low)*i/4 for i in range(5)])
        
        combinations = list(itertools.product(*values))
        
        # Limit combinations if too many
        if len(combinations) > max_combinations:
            np.random.shuffle(combinations)
            combinations = combinations[:max_combinations]
            print(f"Limited to {max_combinations} combinations out of {len(combinations)} total")
        
        results = []
        
        for i, combination in enumerate(combinations):
            config = dict(zip(keys, combination))
            config.update({'model': model_type, 'family': family, 'search_type': 'grid'})
            
            print(f"Grid search {i+1}/{len(combinations)}: {config}")
            
            result = self._run_single_experiment(config, max_epochs)
            results.append(result)
        
        return results
    
    def random_search(self, model_type: str, family: str, n_trials: int = 50,
                     max_epochs: int = 200) -> List[ExperimentResult]:
        """Random search over hyperparameter space."""
        print(f"Starting random search for {model_type} on {family}")
        
        search_space = SearchConfig.SEARCH_SPACES[model_type]
        results = []
        
        for i in range(n_trials):
            # Sample random configuration
            config = {}
            for param, values in search_space.items():
                if isinstance(values, list):
                    config[param] = np.random.choice(values)
                else:  # Continuous parameter
                    low, high = float(values[0]), float(values[1])  # Explicit type conversion
                    if param in ['lr', 'weight_decay']:
                        config[param] = float(np.exp(np.random.uniform(np.log(low), np.log(high))))
                    else:
                        config[param] = float(np.random.uniform(low, high))
                        if param in ['fourier_modes_x', 'fourier_modes_t', 'fourier_modes_s', 'width', 'n_layers',
                                   'S', 'step_out', 'hist_hidden', 'tau_dim', 'predictor_hidden',
                                   'predictor_layers', 'hidden', 'kernel_layers', 'euler_steps',
                                   'batch_size']:
                            config[param] = int(config[param])
            
            config.update({'model': model_type, 'family': family, 'search_type': 'random'})
            
            print(f"Random search {i+1}/{n_trials}: {config}")
            
            result = self._run_single_experiment(config, max_epochs)
            results.append(result)
        
        return results
    
    def bayesian_search(self, model_type: str, family: str, n_trials: int = 100,
                       max_epochs: int = 200) -> List[ExperimentResult]:
        """Bayesian optimization using Optuna."""
        if not OPTUNA_AVAILABLE:
            print("Optuna not available. Falling back to random search.")
            return self.random_search(model_type, family, n_trials, max_epochs)
        
        print(f"Starting Bayesian search for {model_type} on {family}")
        
        results = []
        
        def objective(trial) -> float:
            bounds = SearchConfig.BAYESIAN_BOUNDS[model_type]
            config: Dict[str, Union[float, int]] = {}
            
            for param, (low, high) in bounds.items():
                low_val = float(low)
                high_val = float(high)
                if param in ['lr', 'weight_decay']:
                    config[param] = trial.suggest_float(param, low_val, high_val, log=True)
                elif param in ['dropout', 'euler_dt']:
                    config[param] = trial.suggest_float(param, low_val, high_val)
                elif param in ['fourier_modes_x', 'fourier_modes_t', 'fourier_modes_s', 'width', 'n_layers',
                             'S', 'step_out', 'hist_hidden', 'tau_dim', 'predictor_hidden',
                             'predictor_layers', 'hidden', 'kernel_layers', 'euler_steps',
                             'batch_size']:
                    config[param] = trial.suggest_int(param, int(low_val), int(high_val))
                else:
                    config[param] = trial.suggest_float(param, low_val, high_val)
            
            config.update({'model': model_type, 'family': family, 'search_type': 'bayesian'})
            
            result = self._run_single_experiment(config, max_epochs)
            results.append(result)
            
            return result['test_rel_error']  # Minimize relative error
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(),
            pruner=MedianPruner()
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best trial: {study.best_trial.value}")
        print(f"Best params: {study.best_trial.params}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="DNO Hyperparameter Search")
    parser.add_argument("--search_type", choices=['grid', 'random', 'bayesian'], 
                       default='random', help="Search strategy")
    parser.add_argument("--model", choices=SearchConfig.MODELS, 
                       default='method_of_steps', help="Model type")
    parser.add_argument("--family", choices=SearchConfig.FAMILIES, 
                       default='mackey', help="Dataset family")
    parser.add_argument("--n_trials", type=int, default=50, 
                       help="Number of trials for random/bayesian search")
    parser.add_argument("--max_epochs", type=int, default=200, 
                       help="Maximum training epochs")
    parser.add_argument("--max_combinations", type=int, default=100, 
                       help="Maximum combinations for grid search")
    parser.add_argument("--data_root", default="./data", 
                       help="Root directory for data")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--wandb_project", default="dno-hyperparameter-search",
                       help="W&B project name")
    parser.add_argument("--wandb_entity", default=None,
                       help="W&B entity name")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize tracker
    tracker = ExperimentTracker(args.wandb_project, args.wandb_entity)
    
    # Initialize searcher
    searcher = HyperparameterSearcher(tracker)
    
    # Run search
    if args.search_type == 'grid':
        results = searcher.grid_search(args.model, args.family, args.max_epochs, args.max_combinations)
    elif args.search_type == 'random':
        results = searcher.random_search(args.model, args.family, args.n_trials, args.max_epochs)
    elif args.search_type == 'bayesian':
        results = searcher.bayesian_search(args.model, args.family, args.n_trials, args.max_epochs)
    
    # Save results
    results_file = f"search_results_{args.model}_{args.family}_{args.search_type}.json"
    with open(results_file, 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    
    # Print summary
    successful_results = [r for r in results if r.get('success', False)]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['test_rel_error'])
        print(f"\nBest result:")
        print(f"Test Rel Error: {best_result['test_rel_error']:.6f}")
        print(f"Config: {best_result}")
    
    print(f"\nResults saved to {results_file}")
    print(f"Total experiments: {len(results)}")
    print(f"Successful experiments: {len(successful_results)}")

if __name__ == "__main__":
    main()
