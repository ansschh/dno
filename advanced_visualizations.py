"""
advanced_visualizations.py
==========================
Advanced visualization module for DNO hyperparameter search results.
Generates publication-quality plots including heatmaps, side-by-side comparisons,
and comprehensive analysis visualizations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive plots disabled.")

class AdvancedVisualizer:
    """Advanced visualization suite for DNO hyperparameter search results."""
    
    def __init__(self, results_dir: str = "hyperparameter_results", 
                 output_dir: str = "visualizations"):
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set publication-ready style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Configure matplotlib for high-quality output
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def load_experiment_results(self) -> pd.DataFrame:
        """Load all experiment results from the results directory."""
        results = []
        
        if not os.path.exists(self.results_dir):
            print(f"Results directory {self.results_dir} not found.")
            return pd.DataFrame()
        
        for run_dir in os.listdir(self.results_dir):
            run_path = os.path.join(self.results_dir, run_dir)
            if not os.path.isdir(run_path):
                continue
            
            # Load config and final results
            config_path = os.path.join(run_path, "config.json")
            results_path = os.path.join(run_path, "final_results.json")
            metrics_path = os.path.join(run_path, "metrics.csv")
            
            if os.path.exists(config_path) and os.path.exists(results_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                with open(results_path, 'r') as f:
                    final_results = json.load(f)
                
                # Combine config and results
                result = {**config, **final_results, 'run_id': run_dir}
                
                # Load training metrics if available
                if os.path.exists(metrics_path):
                    metrics_df = pd.read_csv(metrics_path)
                    if not metrics_df.empty:
                        # Add summary statistics from training
                        result['final_train_loss'] = metrics_df['train_loss'].iloc[-1] if 'train_loss' in metrics_df.columns else None
                        result['final_val_loss'] = metrics_df['val_loss'].iloc[-1] if 'val_loss' in metrics_df.columns else None
                        result['min_val_loss'] = metrics_df['val_loss'].min() if 'val_loss' in metrics_df.columns else None
                        result['epochs_trained'] = len(metrics_df)
                
                results.append(result)
        
        return pd.DataFrame(results)
    
    def create_hyperparameter_heatmap(self, df: pd.DataFrame, 
                                    param_x: str, param_y: str, 
                                    metric: str = 'test_rel_error',
                                    model_type: Optional[str] = None,
                                    family: Optional[str] = None) -> plt.Figure:
        """Create a heatmap showing hyperparameter performance."""
        # Filter data if specified
        plot_df = df.copy()
        if model_type:
            plot_df = plot_df[plot_df['model'] == model_type]
        if family:
            plot_df = plot_df[plot_df['family'] == family]
        
        if plot_df.empty or param_x not in plot_df.columns or param_y not in plot_df.columns:
            print(f"Insufficient data for heatmap: {param_x} vs {param_y}")
            return None
        
        # Create pivot table for heatmap
        pivot_data = plot_df.pivot_table(
            values=metric, 
            index=param_y, 
            columns=param_x, 
            aggfunc='mean'
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='viridis_r', 
                   ax=ax, cbar_kws={'label': metric})
        
        title = f'Hyperparameter Heatmap: {param_x} vs {param_y}'
        if model_type:
            title += f' ({model_type})'
        if family:
            title += f' - {family}'
        
        ax.set_title(title)
        ax.set_xlabel(param_x.replace('_', ' ').title())
        ax.set_ylabel(param_y.replace('_', ' ').title())
        
        plt.tight_layout()
        return fig
    
    def create_side_by_side_comparison(self, df: pd.DataFrame, 
                                     models: List[str] = None,
                                     families: List[str] = None,
                                     metric: str = 'test_rel_error') -> plt.Figure:
        """Create side-by-side comparison plots of model performance."""
        if models is None:
            models = df['model'].unique()
        if families is None:
            families = df['family'].unique()
        
        fig, axes = plt.subplots(1, len(families), figsize=(6*len(families), 8))
        if len(families) == 1:
            axes = [axes]
        
        for i, family in enumerate(families):
            family_data = df[df['family'] == family]
            
            # Create box plot for each model
            model_data = []
            model_labels = []
            
            for model in models:
                model_family_data = family_data[family_data['model'] == model]
                if not model_family_data.empty and metric in model_family_data.columns:
                    model_data.append(model_family_data[metric].values)
                    model_labels.append(model)
            
            if model_data:
                axes[i].boxplot(model_data, labels=model_labels)
                axes[i].set_title(f'{family.replace("_", " ").title()} Dataset')
                axes[i].set_ylabel(metric.replace('_', ' ').title())
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(f'Model Performance Comparison: {metric.replace("_", " ").title()}')
        plt.tight_layout()
        return fig
    
    def create_convergence_plots(self, run_ids: List[str] = None, 
                               max_runs: int = 10) -> plt.Figure:
        """Create convergence plots showing training progress."""
        if run_ids is None:
            # Get all available run IDs
            run_ids = [d for d in os.listdir(self.results_dir) 
                      if os.path.isdir(os.path.join(self.results_dir, d))][:max_runs]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))
        
        for i, run_id in enumerate(run_ids):
            metrics_path = os.path.join(self.results_dir, run_id, "metrics.csv")
            config_path = os.path.join(self.results_dir, run_id, "config.json")
            
            if not os.path.exists(metrics_path) or not os.path.exists(config_path):
                continue
            
            metrics_df = pd.read_csv(metrics_path)
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            label = f"{config.get('model', 'unknown')}_{config.get('family', 'unknown')}"
            color = colors[i % len(colors)]
            
            # Training loss
            if 'train_loss' in metrics_df.columns:
                axes[0].plot(metrics_df['step'], metrics_df['train_loss'], 
                           label=label, color=color, alpha=0.7)
            
            # Validation loss
            if 'val_loss' in metrics_df.columns:
                axes[1].plot(metrics_df['step'], metrics_df['val_loss'], 
                           label=label, color=color, alpha=0.7)
            
            # Relative error
            if 'val_rel_error' in metrics_df.columns:
                axes[2].plot(metrics_df['step'], metrics_df['val_rel_error'], 
                           label=label, color=color, alpha=0.7)
            
            # Learning rate (if available)
            if 'learning_rate' in metrics_df.columns:
                axes[3].plot(metrics_df['step'], metrics_df['learning_rate'], 
                           label=label, color=color, alpha=0.7)
        
        # Configure subplots
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('Validation Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_title('Validation Relative Error')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Relative Error')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        axes[3].set_title('Learning Rate Schedule')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Learning Rate')
        axes[3].set_yscale('log')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.suptitle('Training Convergence Analysis')
        plt.tight_layout()
        return fig
    
    def create_parameter_importance_plot(self, df: pd.DataFrame, 
                                       metric: str = 'test_rel_error',
                                       top_k: int = 10) -> plt.Figure:
        """Create a plot showing parameter importance based on correlation with performance."""
        # Select numeric columns (hyperparameters)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        param_cols = [col for col in numeric_cols if col not in [
            'test_rel_error', 'test_mse', 'train_loss', 'val_loss', 
            'final_train_loss', 'final_val_loss', 'min_val_loss', 'epochs_trained'
        ]]
        
        if not param_cols or metric not in df.columns:
            print("Insufficient data for parameter importance analysis")
            return None
        
        # Calculate correlations
        correlations = []
        for param in param_cols:
            if df[param].nunique() > 1:  # Only consider parameters that vary
                corr = abs(df[param].corr(df[metric]))
                if not np.isnan(corr):
                    correlations.append((param, corr))
        
        # Sort by importance and take top k
        correlations.sort(key=lambda x: x[1], reverse=True)
        correlations = correlations[:top_k]
        
        if not correlations:
            print("No significant correlations found")
            return None
        
        # Create plot
        params, importances = zip(*correlations)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(params)), importances)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels([p.replace('_', ' ').title() for p in params])
        ax.set_xlabel(f'Absolute Correlation with {metric.replace("_", " ").title()}')
        ax.set_title('Hyperparameter Importance Analysis')
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', va='center')
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        return fig
    
    def create_interactive_scatter_plot(self, df: pd.DataFrame,
                                      x_param: str, y_param: str,
                                      color_param: str = 'test_rel_error',
                                      size_param: Optional[str] = None) -> Optional[str]:
        """Create an interactive scatter plot using Plotly."""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available for interactive plots")
            return None
        
        if not all(param in df.columns for param in [x_param, y_param, color_param]):
            print(f"Missing columns for interactive plot")
            return None
        
        # Create hover text
        hover_text = []
        for _, row in df.iterrows():
            text = f"Model: {row.get('model', 'N/A')}<br>"
            text += f"Family: {row.get('family', 'N/A')}<br>"
            text += f"{color_param}: {row.get(color_param, 'N/A'):.4f}<br>"
            text += f"Run ID: {row.get('run_id', 'N/A')}"
            hover_text.append(text)
        
        # Create scatter plot
        fig = px.scatter(
            df, x=x_param, y=y_param, color=color_param,
            size=size_param if size_param and size_param in df.columns else None,
            hover_name='run_id',
            title=f'Interactive Hyperparameter Analysis: {x_param} vs {y_param}',
            labels={
                x_param: x_param.replace('_', ' ').title(),
                y_param: y_param.replace('_', ' ').title(),
                color_param: color_param.replace('_', ' ').title()
            }
        )
        
        fig.update_traces(hovertemplate='<b>%{hovertext}</b><extra></extra>',
                         hovertext=hover_text)
        
        # Save as HTML
        output_path = os.path.join(self.output_dir, f'interactive_{x_param}_vs_{y_param}.html')
        fig.write_html(output_path)
        return output_path
    
    def generate_comprehensive_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a comprehensive visualization report."""
        report = {
            'summary': {},
            'figures': {},
            'interactive_plots': []
        }
        
        if df.empty:
            print("No data available for visualization report")
            return report
        
        # Summary statistics
        report['summary'] = {
            'total_experiments': len(df),
            'models_tested': df['model'].nunique() if 'model' in df.columns else 0,
            'families_tested': df['family'].nunique() if 'family' in df.columns else 0,
            'best_performance': df['test_rel_error'].min() if 'test_rel_error' in df.columns else None,
            'worst_performance': df['test_rel_error'].max() if 'test_rel_error' in df.columns else None
        }
        
        print(f"Generating comprehensive report for {len(df)} experiments...")
        
        # 1. Side-by-side model comparison
        if 'model' in df.columns and 'family' in df.columns:
            fig = self.create_side_by_side_comparison(df)
            if fig:
                fig_path = os.path.join(self.output_dir, 'model_comparison.png')
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                report['figures']['model_comparison'] = fig_path
                plt.close(fig)
        
        # 2. Parameter importance
        fig = self.create_parameter_importance_plot(df)
        if fig:
            fig_path = os.path.join(self.output_dir, 'parameter_importance.png')
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            report['figures']['parameter_importance'] = fig_path
            plt.close(fig)
        
        # 3. Convergence plots
        fig = self.create_convergence_plots()
        if fig:
            fig_path = os.path.join(self.output_dir, 'convergence_analysis.png')
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            report['figures']['convergence_analysis'] = fig_path
            plt.close(fig)
        
        # 4. Hyperparameter heatmaps
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        param_cols = [col for col in numeric_cols if col not in [
            'test_rel_error', 'test_mse', 'train_loss', 'val_loss'
        ]]
        
        # Generate heatmaps for important parameter pairs
        important_pairs = [
            ('learning_rate', 'batch_size'),
            ('fourier_modes_x', 'fourier_modes_t'),
            ('width', 'n_layers'),
            ('dropout', 'weight_decay')
        ]
        
        for i, (param_x, param_y) in enumerate(important_pairs):
            if param_x in param_cols and param_y in param_cols:
                fig = self.create_hyperparameter_heatmap(df, param_x, param_y)
                if fig:
                    fig_path = os.path.join(self.output_dir, f'heatmap_{param_x}_vs_{param_y}.png')
                    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                    report['figures'][f'heatmap_{i}'] = fig_path
                    plt.close(fig)
        
        # 5. Interactive plots (if Plotly available)
        if PLOTLY_AVAILABLE and len(param_cols) >= 2:
            for param_x, param_y in important_pairs[:2]:  # Limit to avoid too many files
                if param_x in param_cols and param_y in param_cols:
                    html_path = self.create_interactive_scatter_plot(df, param_x, param_y)
                    if html_path:
                        report['interactive_plots'].append(html_path)
        
        # Save report summary
        report_path = os.path.join(self.output_dir, 'visualization_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Comprehensive report generated: {len(report['figures'])} static plots, "
              f"{len(report['interactive_plots'])} interactive plots")
        
        return report

def main():
    """Generate visualizations for all available hyperparameter search results."""
    visualizer = AdvancedVisualizer()
    
    # Load all experiment results
    df = visualizer.load_experiment_results()
    
    if df.empty:
        print("No experiment results found. Run hyperparameter search first.")
        return
    
    print(f"Loaded {len(df)} experiment results")
    print(f"Models: {df['model'].unique() if 'model' in df.columns else 'N/A'}")
    print(f"Families: {df['family'].unique() if 'family' in df.columns else 'N/A'}")
    
    # Generate comprehensive report
    report = visualizer.generate_comprehensive_report(df)
    
    print("\nVisualization Report Summary:")
    print(f"- Total experiments: {report['summary']['total_experiments']}")
    print(f"- Models tested: {report['summary']['models_tested']}")
    print(f"- Families tested: {report['summary']['families_tested']}")
    if report['summary']['best_performance']:
        print(f"- Best performance: {report['summary']['best_performance']:.6f}")
    
    print(f"\nGenerated visualizations saved to: {visualizer.output_dir}/")

if __name__ == "__main__":
    main()
