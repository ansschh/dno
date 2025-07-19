"""
experiment_tracker.py
=====================
Comprehensive experiment tracking with W&B integration for DNO research.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
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
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: W&B not available. Advanced logging disabled.")

class ExperimentTracker:
    """Comprehensive experiment tracking with W&B integration."""
    
    def __init__(self, project_name: str = "dno-hyperparameter-search", 
                 entity: Optional[str] = None):
        self.project_name = project_name
        self.entity = entity
        self.results_dir = "hyperparameter_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize W&B if available
        self.wandb_enabled = WANDB_AVAILABLE
        if self.wandb_enabled:
            try:
                wandb.login()
                print("✓ W&B authentication successful")
            except Exception as e:
                print(f"W&B login failed: {e}")
                self.wandb_enabled = False
    
    def start_run(self, config: Dict[str, Any], run_name: Optional[str] = None):
        """Start a new experiment run."""
        if self.wandb_enabled:
            wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=run_name,
                config=config,
                reinit=True
            )
        
        # Local logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{config.get('model', 'unknown')}_{config.get('family', 'all')}_{timestamp}"
        self.current_run_dir = os.path.join(self.results_dir, run_id)
        os.makedirs(self.current_run_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.current_run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        return run_id
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to W&B and local files."""
        if self.wandb_enabled:
            wandb.log(metrics, step=step)
        
        # Local CSV logging
        metrics_file = os.path.join(self.current_run_dir, "metrics.csv")
        df_new = pd.DataFrame([{**metrics, 'step': step}])
        
        if os.path.exists(metrics_file):
            df_existing = pd.read_csv(metrics_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        df_combined.to_csv(metrics_file, index=False)
    
    def log_artifacts(self, artifacts: Dict[str, Any]):
        """Log artifacts (plots, models, etc.)."""
        for name, artifact in artifacts.items():
            artifact_path = os.path.join(self.current_run_dir, f"{name}")
            
            if isinstance(artifact, plt.Figure):
                artifact.savefig(f"{artifact_path}.png", dpi=300, bbox_inches='tight')
                if self.wandb_enabled:
                    wandb.log({name: wandb.Image(f"{artifact_path}.png")})
                plt.close(artifact)  # Free memory
            
            elif isinstance(artifact, dict):
                with open(f"{artifact_path}.json", "w") as f:
                    json.dump(artifact, f, indent=2)
                if self.wandb_enabled:
                    wandb.save(f"{artifact_path}.json")
            
            elif hasattr(artifact, 'state_dict'):  # PyTorch model
                import torch
                torch.save(artifact.state_dict(), f"{artifact_path}.pt")
                if self.wandb_enabled:
                    wandb.save(f"{artifact_path}.pt")
            
            elif isinstance(artifact, str) and artifact.endswith('.html'):
                # Handle HTML files (interactive plots)
                if self.wandb_enabled:
                    wandb.save(artifact)
            
            elif isinstance(artifact, np.ndarray):
                # Handle numpy arrays (predictions, etc.)
                np.save(f"{artifact_path}.npy", artifact)
                if self.wandb_enabled:
                    wandb.save(f"{artifact_path}.npy")
    
    def finish_run(self, final_metrics: Dict[str, float]):
        """Finish the current run."""
        if self.wandb_enabled:
            wandb.log(final_metrics)
            wandb.finish()
        
        # Save final results
        with open(os.path.join(self.current_run_dir, "final_results.json"), "w") as f:
            json.dump(convert_numpy_types(final_metrics), f, indent=2)
