"""
hyperparameter_config.py
========================
Configuration and search spaces for DNO hyperparameter optimization.
"""

class SearchConfig:
    """Configuration for hyperparameter search."""
    
    # Model architectures to search
    MODELS = ['stacked_history', 'method_of_steps', 'memory_kernel']
    
    # Dataset families
    FAMILIES = ['mackey', 'delayed_logistic', 'neutral', 'reaction_diffusion']
    
    # Search spaces for each model
    SEARCH_SPACES = {
        'stacked_history': {
            'fourier_modes_x': [8, 16, 32, 64],
            'fourier_modes_t': [8, 16, 32, 64], 
            'width': [32, 64, 128, 256],
            'n_layers': [2, 3, 4, 5, 6],
            'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [8, 16, 32, 64],
            'dropout': [0.0, 0.1, 0.2, 0.3],
            'weight_decay': [0.0, 1e-6, 1e-5, 1e-4]
        },
        'method_of_steps': {
            'S': [16, 32, 64, 128],
            'step_out': [8, 16, 32, 64],
            'hist_hidden': [64, 128, 256, 512],
            'tau_dim': [8, 16, 32, 64],
            'predictor_hidden': [128, 256, 512, 1024],
            'predictor_layers': [2, 3, 4, 5],
            'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [8, 16, 32, 64],
            'dropout': [0.0, 0.1, 0.2, 0.3],
            'weight_decay': [0.0, 1e-6, 1e-5, 1e-4]
        },
        'memory_kernel': {
            'S': [16, 32, 64, 128],
            'hidden': [64, 128, 256, 512],
            'kernel_layers': [2, 3, 4, 5],
            'euler_dt': [0.01, 0.02, 0.05, 0.1],
            'euler_steps': [50, 100, 200, 500],
            'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [8, 16, 32, 64],
            'dropout': [0.0, 0.1, 0.2, 0.3],
            'weight_decay': [0.0, 1e-6, 1e-5, 1e-4]
        }
    }
    
    # Bayesian optimization bounds
    BAYESIAN_BOUNDS = {
        'stacked_history': {
            'fourier_modes_x': (4, 128),
            'fourier_modes_t': (4, 128),
            'width': (16, 512),
            'n_layers': (2, 8),
            'lr': (1e-6, 1e-2),
            'batch_size': (4, 128),
            'dropout': (0.0, 0.5),
            'weight_decay': (0.0, 1e-3)
        },
        'method_of_steps': {
            'S': (8, 256),
            'step_out': (4, 128),
            'hist_hidden': (32, 1024),
            'tau_dim': (4, 128),
            'predictor_hidden': (64, 2048),
            'predictor_layers': (2, 8),
            'lr': (1e-6, 1e-2),
            'batch_size': (4, 128),
            'dropout': (0.0, 0.5),
            'weight_decay': (0.0, 1e-3)
        },
        'memory_kernel': {
            'S': (8, 256),
            'hidden': (32, 1024),
            'kernel_layers': (2, 8),
            'euler_dt': (0.001, 0.2),
            'euler_steps': (20, 1000),
            'lr': (1e-6, 1e-2),
            'batch_size': (4, 128),
            'dropout': (0.0, 0.5),
            'weight_decay': (0.0, 1e-3)
        }
    }
