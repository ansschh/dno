"""
model_trainer.py
===============
Robust model training with early stopping and comprehensive evaluation for DNO models.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any, Optional

from common_py34 import relative_l2
from stacked_history import StackedHistoryFNO
from method_of_steps import MethodOfStepsModel
from memory_kernel import MemoryKernelModel
from experiment_tracker import ExperimentTracker

class ModelTrainer:
    """Robust model training with early stopping and comprehensive evaluation."""
    
    def __init__(self, device: str = "cuda", patience: int = 20, min_delta: float = 1e-6):
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
    
    def create_model(self, model_type: str, config: Dict[str, Any], 
                    input_channels: int, output_channels: int) -> nn.Module:
        """Create model instance based on type and config."""
        if model_type == 'stacked_history':
            # Ensure fourier_modes_x is adjusted for 1D data
            fourier_modes_x = config.get('fourier_modes_x', 16)
            if output_channels == 1:
                fourier_modes_x = min(fourier_modes_x, 1)
            
            # Determine X dimension (spatial width)
            # For reaction_diffusion (20D output), use output_channels as spatial dimension
            # For 1D systems, use X=1
            X = output_channels if output_channels > 1 else 1
            
            model = StackedHistoryFNO(
                in_channels=output_channels,
                out_channels=output_channels,
                S=config.get('S', 32),
                X=X,
                fourier_modes_s=config.get('fourier_modes_s', config.get('fourier_modes_t', 16)),
                fourier_modes_x=fourier_modes_x,
                hidden_channels=config.get('width', 64),
                n_layers=config.get('n_layers', 4),
                dropout=config.get('dropout', 0.1)
            )
        
        elif model_type == 'method_of_steps':
            model = MethodOfStepsModel(
                S=config.get('S', 32),
                step_out=config.get('step_out', 16),
                roll_steps=config.get('roll_steps', 5),
                in_channels=output_channels,
                out_channels=output_channels,
                hist_hidden=config.get('hist_hidden', 64),
                hist_layers=config.get('hist_layers', 3),
                tau_dim=config.get('tau_dim', 16),
                predictor_hidden=config.get('predictor_hidden', 128)
            )
        
        elif model_type == 'memory_kernel':
            model = MemoryKernelModel(
                S=config.get('S', 32),
                channels=output_channels,
                hidden=config.get('hidden', 64)
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def train_and_evaluate(self, model_type: str, config: Dict[str, Any], 
                          train_loader: DataLoader, val_loader: DataLoader,
                          test_loader: DataLoader, max_epochs: int = 200,
                          tracker: Optional[ExperimentTracker] = None) -> Dict[str, float]:
        """Train model with early stopping and comprehensive evaluation."""
        
        # Determine input/output dimensions from data
        sample_batch = next(iter(train_loader))
        y_sample = sample_batch["y"]
        if y_sample.ndim == 3:
            _, _, output_channels = y_sample.shape
        else:
            output_channels = 1
        
        hist_sample = sample_batch["hist"]
        if hist_sample.ndim == 3:
            _, _, input_channels = hist_sample.shape
        else:
            input_channels = 1
        
        # Create model
        model = self.create_model(model_type, config, input_channels, output_channels)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        train_losses = []
        val_losses = []
        val_rel_errors = []
        
        for epoch in range(max_epochs):
            # Training
            model.train()
            epoch_train_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                hist = batch["hist"].to(self.device)
                y = batch["y"].to(self.device)
                
                if model_type == 'stacked_history':
                    target_len = min(model.S, y.shape[1])
                    target = y[:, :target_len, :]
                    pred = model(hist)[:, :target_len, :]
                
                elif model_type == 'method_of_steps':
                    tau = batch["tau"].to(self.device)
                    target_len = min(model.step_out, y.shape[1])
                    target = y[:, :target_len, :]
                    pred = model(hist, tau)[:, :target_len, :]
                
                elif model_type == 'memory_kernel':
                    B, T, C = y.shape
                    hist_processed = model.init_history(hist)
                    steps = min(config.get('euler_steps', 100), T)
                    pred = model.euler_roll(hist_processed, steps, config.get('euler_dt', 0.05))
                    target = y[:, :steps, :]
                
                loss = torch.mean((pred - target) ** 2)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            val_loss, val_rel = self.evaluate_model(model, model_type, val_loader, config)
            val_losses.append(val_loss)
            val_rel_errors.append(val_rel)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Log metrics
            metrics = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_rel_error': val_rel,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'patience_counter': patience_counter
            }
            
            if tracker:
                tracker.log_metrics(metrics, epoch)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, Val Rel: {val_rel:.6f}")
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model for final evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final comprehensive evaluation
        test_loss, test_rel = self.evaluate_model(model, model_type, test_loader, config)
        
        # Multi-step rollout evaluation (if applicable)
        rollout_metrics = self.evaluate_rollout(model, model_type, test_loader, config)
        
        # Stability analysis
        stability_metrics = self.analyze_stability(model, model_type, test_loader, config)
        
        final_results = {
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'test_rel_error': test_rel,
            'final_epoch': epoch,
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            **rollout_metrics,
            **stability_metrics
        }
        
        return final_results
    
    def evaluate_model(self, model: nn.Module, model_type: str, 
                      loader: DataLoader, config: Dict[str, Any]) -> Tuple[float, float]:
        """Evaluate model on given data loader."""
        model.eval()
        total_loss = 0.0
        rel_errors = []
        num_samples = 0
        
        with torch.no_grad():
            for batch in loader:
                hist = batch["hist"].to(self.device)
                y = batch["y"].to(self.device)
                B = y.shape[0]
                
                if model_type == 'stacked_history':
                    target_len = min(model.S, y.shape[1])
                    target = y[:, :target_len, :]
                    pred = model(hist)[:, :target_len, :]
                
                elif model_type == 'method_of_steps':
                    tau = batch["tau"].to(self.device)
                    target_len = min(model.step_out, y.shape[1])
                    target = y[:, :target_len, :]
                    pred = model(hist, tau)[:, :target_len, :]
                
                elif model_type == 'memory_kernel':
                    B, T, C = y.shape
                    hist_processed = model.init_history(hist)
                    steps = min(config.get('euler_steps', 100), T)
                    pred = model.euler_roll(hist_processed, steps, config.get('euler_dt', 0.05))
                    target = y[:, :steps, :]
                
                loss = torch.mean((pred - target) ** 2)
                total_loss += loss.item() * B
                
                rel_error = relative_l2(pred, target)
                if isinstance(rel_error, torch.Tensor):
                    rel_error = rel_error.cpu().item()
                rel_errors.append(rel_error)
                
                num_samples += B
        
        avg_loss = total_loss / num_samples
        avg_rel_error = np.mean(rel_errors)
        
        return avg_loss, avg_rel_error
    
    def evaluate_rollout(self, model: nn.Module, model_type: str,
                        loader: DataLoader, config: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate multi-step rollout performance."""
        if model_type != 'method_of_steps':
            return {'rollout_loss': 0.0, 'rollout_rel_error': 0.0}
        
        model.eval()
        rollout_losses = []
        rollout_rel_errors = []
        
        with torch.no_grad():
            for batch in loader:
                hist = batch["hist"].to(self.device)
                tau = batch["tau"].to(self.device)
                y = batch["y"].to(self.device)
                B, T, C = y.shape
                
                # Multi-step rollout
                steps = min(model.roll_steps, T // model.step_out)
                if steps > 1:
                    pred_full = model.rollout(hist, tau, steps)
                    min_len = min(pred_full.shape[1], y.shape[1])
                    pred = pred_full[:, :min_len, :]
                    target = y[:, :min_len, :]
                    
                    loss = torch.mean((pred - target) ** 2).item()
                    rel_error = relative_l2(pred, target)
                    if isinstance(rel_error, torch.Tensor):
                        rel_error = rel_error.cpu().item()
                    
                    rollout_losses.append(loss)
                    rollout_rel_errors.append(rel_error)
        
        return {
            'rollout_loss': np.mean(rollout_losses) if rollout_losses else 0.0,
            'rollout_rel_error': np.mean(rollout_rel_errors) if rollout_rel_errors else 0.0
        }
    
    def analyze_stability(self, model: nn.Module, model_type: str,
                         loader: DataLoader, config: Dict[str, Any]) -> Dict[str, float]:
        """Analyze model stability and robustness."""
        model.eval()
        
        # Prediction variance analysis
        prediction_vars = []
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 10:  # Limit analysis to first 10 batches
                    break
                
                hist = batch["hist"].to(self.device)
                y = batch["y"].to(self.device)
                
                # Add small noise to inputs and measure prediction variance
                noise_levels = [0.0, 0.01, 0.05, 0.1]
                predictions = []
                
                for noise_level in noise_levels:
                    hist_noisy = hist + torch.randn_like(hist) * noise_level
                    
                    if model_type == 'stacked_history':
                        pred = model(hist_noisy)
                    elif model_type == 'method_of_steps':
                        tau = batch["tau"].to(self.device)
                        pred = model(hist_noisy, tau)
                    elif model_type == 'memory_kernel':
                        hist_processed = model.init_history(hist_noisy)
                        pred = model.euler_roll(hist_processed, 50, config.get('euler_dt', 0.05))
                    
                    predictions.append(pred.cpu().numpy())
                
                # Calculate prediction variance
                pred_var = np.var(predictions, axis=0).mean()
                prediction_vars.append(pred_var)
        
        return {
            'prediction_stability': np.mean(prediction_vars),
            'stability_score': 1.0 / (1.0 + np.mean(prediction_vars))  # Higher is more stable
        }
