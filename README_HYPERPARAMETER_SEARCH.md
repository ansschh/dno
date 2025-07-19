# Delay Neural Operator (DNO) Hyperparameter Search Pipeline

## Overview

This is a production-ready hyperparameter search and visualization pipeline for Delay Neural Operator (DNO) models. The pipeline supports three DNO variants (stacked_history, method_of_steps, memory_kernel) across four DDE dataset families with comprehensive experiment tracking and advanced visualizations.

## Features

### 🔍 Hyperparameter Search Strategies
- **Grid Search**: Exhaustive search over parameter combinations
- **Random Search**: Efficient random sampling of parameter space
- **Bayesian Optimization**: Intelligent search using Optuna with TPE sampler

### 📊 Advanced Visualizations
- **Hyperparameter Heatmaps**: 2D performance landscapes
- **Side-by-side Model Comparisons**: Box plots across datasets
- **Training Convergence Analysis**: Loss and metric evolution
- **Parameter Importance Analysis**: Correlation-based ranking
- **Interactive Plots**: Plotly-based exploration tools

### 🚀 Production Features
- **Weights & Biases Integration**: Comprehensive experiment tracking
- **Early Stopping**: Prevent overfitting with patience-based stopping
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Gradient Clipping**: Stable training for large models
- **Multi-step Rollout Evaluation**: Long-term prediction assessment
- **Stability Analysis**: Robustness testing with noise injection

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Login to Weights & Biases (optional but recommended)
wandb login
```

### 2. Basic Usage

```bash
# Run Bayesian hyperparameter search (recommended)
python hyperparameter_search.py --search_type bayesian --n_trials 100 --max_epochs 200

# Run random search for faster results
python hyperparameter_search.py --search_type random --n_trials 50 --max_epochs 100

# Run grid search (for small parameter spaces)
python hyperparameter_search.py --search_type grid --max_combinations 50
```

### 3. Generate Visualizations

```bash
# Generate comprehensive visualization report
python advanced_visualizations.py
```

## Configuration

### Search Configuration (`hyperparameter_config.py`)

The search spaces are defined for each model type:

```python
# Example search space for method_of_steps model
SEARCH_SPACES = {
    'method_of_steps': {
        'learning_rate': (1e-5, 1e-2, 'log'),
        'batch_size': [16, 32, 64, 128],
        'fourier_modes_x': [8, 16, 32, 64],
        'fourier_modes_t': [8, 16, 32, 64],
        'width': [32, 64, 128, 256],
        'n_layers': [2, 3, 4, 5, 6],
        'dropout': (0.0, 0.3, 'uniform'),
        'weight_decay': (1e-6, 1e-3, 'log')
    }
}
```

### Command Line Arguments

```bash
python hyperparameter_search.py --help

Arguments:
  --search_type {grid,random,bayesian}  Search strategy (default: random)
  --model {stacked_history,method_of_steps,memory_kernel}  Model type
  --family {mackey,delayed_logistic,neutral,reaction_diffusion}  Dataset family
  --n_trials INT                        Number of trials for random/bayesian search
  --max_epochs INT                      Maximum training epochs (default: 200)
  --data_root PATH                      Root directory for data (default: ./data)
  --wandb_project STR                   W&B project name
  --wandb_entity STR                    W&B entity name
```

## Architecture

### Core Components

1. **`hyperparameter_search.py`**: Main search orchestrator
2. **`hyperparameter_config.py`**: Search space definitions
3. **`experiment_tracker.py`**: W&B integration and logging
4. **`model_trainer.py`**: Training loop with advanced features
5. **`advanced_visualizations.py`**: Publication-quality plots

### Model Variants

1. **Stacked History**: Treats history as additional spatial dimension
2. **Method of Steps**: Maps (history, delay) → next solution window
3. **Memory Kernel**: Uses learnable kernel weights in frequency space

### Dataset Families

1. **Mackey-Glass**: Chaotic dynamics, variable time steps
2. **Delayed Logistic**: Extreme dynamics, variable time steps  
3. **Neutral DDE**: Delayed derivatives, fixed 500 time steps
4. **Reaction-Diffusion**: Spatial-temporal, 20D output, fixed 500 time steps

## Results and Analysis

### Output Structure

```
hyperparameter_results/
├── {model}_{family}_{timestamp}/
│   ├── config.json              # Experiment configuration
│   ├── metrics.csv              # Training metrics per epoch
│   ├── final_results.json       # Final test performance
│   └── artifacts/               # Plots, model checkpoints
└── ...

visualizations/
├── model_comparison.png         # Side-by-side performance
├── parameter_importance.png     # Correlation analysis
├── convergence_analysis.png     # Training curves
├── heatmap_*.png               # Parameter interaction maps
├── interactive_*.html          # Interactive exploration
└── visualization_report.json   # Summary statistics
```

### Key Metrics

- **Test Relative L2 Error**: Primary optimization target
- **Multi-step Rollout Error**: Long-term prediction quality
- **Stability Score**: Robustness to input perturbations
- **Training Efficiency**: Convergence speed and final loss

## Advanced Usage

### Custom Search Spaces

Modify `hyperparameter_config.py` to add new parameters or adjust ranges:

```python
# Add new parameter
'new_param': (min_val, max_val, 'uniform'),  # or 'log'
'categorical_param': ['option1', 'option2', 'option3']
```

### Multi-Objective Optimization

The pipeline can be extended for multi-objective optimization:

```python
# In objective function, return tuple of metrics
return test_rel_error, rollout_error, stability_score
```

### Custom Visualizations

Add new visualization functions to `advanced_visualizations.py`:

```python
def create_custom_plot(self, df: pd.DataFrame) -> plt.Figure:
    # Your custom visualization logic
    pass
```

## Performance Guidelines

### Computational Requirements

- **Memory**: 8-16 GB RAM recommended
- **GPU**: CUDA-compatible GPU with 6+ GB VRAM
- **Storage**: 10+ GB for results and artifacts
- **Time**: 1-24 hours depending on search strategy and epochs

### Optimization Tips

1. **Start with Random Search**: Quick exploration of parameter space
2. **Use Bayesian Search**: For thorough optimization with limited budget
3. **Enable Early Stopping**: Prevent overfitting and save time
4. **Monitor W&B Dashboard**: Real-time experiment tracking
5. **Generate Visualizations**: Identify patterns and best practices

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model width
2. **W&B Login Failed**: Run `wandb login` or set `--wandb_entity None`
3. **Data Not Found**: Check `--data_root` path and file structure
4. **Optuna Import Error**: Install with `pip install optuna`

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{dno_hyperparameter_search,
  title={Delay Neural Operator Hyperparameter Search Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/dno-hyperparameter-search}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review W&B experiment logs
- Examine visualization reports for insights
