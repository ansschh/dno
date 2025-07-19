"""
evaluation_visualizer.py
========================
Visualization and analysis tools for comprehensive DNO evaluation results.
Creates publication-ready plots and comparative analysis across models and datasets.

Usage:
    python evaluation_visualizer.py --results comprehensive_evaluation_all_combinations.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any
import argparse
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EvaluationVisualizer:
    """Visualization and analysis for comprehensive DNO evaluation results."""
    
    def __init__(self, results_file: str):
        """Initialize with evaluation results."""
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.models = ['stacked_history', 'method_of_steps', 'memory_kernel']
        self.families = ['mackey', 'delayed_logistic', 'neutral', 'reaction_diffusion']
        
        # Create output directory
        os.makedirs('evaluation_plots', exist_ok=True)
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with all key metrics."""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Accuracy Heatmap
        plt.subplot(3, 4, 1)
        self._plot_accuracy_heatmap('rel_l2_error', 'Relative L2 Error')
        
        plt.subplot(3, 4, 2)
        self._plot_accuracy_heatmap('correlation', 'Correlation')
        
        # 2. Stability Metrics
        plt.subplot(3, 4, 3)
        self._plot_stability_comparison()
        
        # 3. Efficiency Comparison
        plt.subplot(3, 4, 4)
        self._plot_efficiency_comparison()
        
        # 4. Model Performance Radar
        plt.subplot(3, 4, 5)
        self._plot_performance_radar('mackey')
        
        plt.subplot(3, 4, 6)
        self._plot_performance_radar('reaction_diffusion')
        
        # 5. Robustness Analysis
        plt.subplot(3, 4, 7)
        self._plot_robustness_analysis()
        
        # 6. Training Time vs Accuracy
        plt.subplot(3, 4, 8)
        self._plot_training_time_vs_accuracy()
        
        # 7. Parameter Count vs Performance
        plt.subplot(3, 4, 9)
        self._plot_params_vs_performance()
        
        # 8. Temporal Dynamics Quality
        plt.subplot(3, 4, 10)
        self._plot_temporal_dynamics()
        
        # 9. Cross-Dataset Performance
        plt.subplot(3, 4, 11)
        self._plot_cross_dataset_performance()
        
        # 10. Overall Model Ranking
        plt.subplot(3, 4, 12)
        self._plot_model_ranking()
        
        plt.tight_layout()
        plt.savefig('evaluation_plots/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.savefig('evaluation_plots/comprehensive_dashboard.pdf', bbox_inches='tight')
        print("📊 Comprehensive dashboard saved to evaluation_plots/comprehensive_dashboard.png")
        plt.show()
    
    def _plot_accuracy_heatmap(self, metric: str, title: str):
        """Plot accuracy heatmap for a specific metric."""
        data = np.zeros((len(self.models), len(self.families)))
        
        for i, model in enumerate(self.models):
            for j, family in enumerate(self.families):
                key = f"{model}_{family}"
                if key in self.results:
                    data[i, j] = self.results[key]['accuracy'][metric]
                else:
                    data[i, j] = np.nan
        
        sns.heatmap(data, 
                   xticklabels=self.families, 
                   yticklabels=self.models,
                   annot=True, 
                   fmt='.4f',
                   cmap='RdYlBu_r' if 'error' in metric else 'RdYlBu',
                   cbar_kws={'label': title})
        plt.title(title)
        plt.xlabel('Dataset Family')
        plt.ylabel('Model Type')
    
    def _plot_stability_comparison(self):
        """Plot stability metrics comparison."""
        metrics = ['energy_drift', 'lyapunov_estimate', 'mean_divergence']
        
        data = []
        for model in self.models:
            for family in self.families:
                key = f"{model}_{family}"
                if key in self.results:
                    for metric in metrics:
                        data.append({
                            'Model': model,
                            'Family': family,
                            'Metric': metric,
                            'Value': abs(self.results[key]['stability'][metric])  # Use absolute value
                        })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar plot
        pivot_df = df.pivot_table(values='Value', index=['Model', 'Family'], columns='Metric')
        pivot_df.plot(kind='bar', stacked=False, ax=plt.gca())
        plt.title('Stability Metrics Comparison')
        plt.xlabel('Model-Family')
        plt.ylabel('Stability Score (log scale)')
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.legend(title='Stability Metrics')
    
    def _plot_efficiency_comparison(self):
        """Plot computational efficiency comparison."""
        param_counts = []
        inference_times = []
        labels = []
        
        for model in self.models:
            for family in self.families:
                key = f"{model}_{family}"
                if key in self.results:
                    param_counts.append(self.results[key]['efficiency']['param_count'])
                    inference_times.append(self.results[key]['efficiency']['inference_time_ms'])
                    labels.append(f"{model}\n{family}")
        
        # Scatter plot: parameters vs inference time
        plt.scatter(param_counts, inference_times, s=100, alpha=0.7)
        
        for i, label in enumerate(labels):
            plt.annotate(label, (param_counts[i], inference_times[i]), 
                        fontsize=8, ha='center', va='bottom')
        
        plt.xlabel('Parameter Count')
        plt.ylabel('Inference Time (ms)')
        plt.title('Efficiency: Parameters vs Inference Time')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    def _plot_performance_radar(self, family: str):
        """Plot radar chart for model performance on a specific family."""
        # Define metrics for radar chart
        metrics = ['accuracy', 'stability', 'efficiency', 'robustness', 'temporal']
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig_ax = plt.gca()
        
        for model in self.models:
            key = f"{model}_{family}"
            if key in self.results:
                values = []
                
                # Normalize metrics to 0-1 scale
                acc = 1 - min(self.results[key]['accuracy']['rel_l2_error'], 1.0)  # Lower is better
                stab = max(0, 1 - abs(self.results[key]['stability']['lyapunov_estimate']))  # Lower is better
                eff = 1 / (1 + self.results[key]['efficiency']['inference_time_ms'] / 100)  # Lower is better
                rob = 1 - min(self.results[key]['robustness']['avg_noise_sensitivity'], 1.0)  # Lower is better
                temp = self.results[key]['temporal_dynamics']['frequency_correlation']  # Higher is better
                
                values = [acc, stab, eff, rob, temp]
                values += values[:1]  # Complete the circle
                
                fig_ax.plot(angles, values, 'o-', linewidth=2, label=model)
                fig_ax.fill(angles, values, alpha=0.25)
        
        fig_ax.set_xticks(angles[:-1])
        fig_ax.set_xticklabels(metrics)
        fig_ax.set_ylim(0, 1)
        fig_ax.set_title(f'Performance Radar: {family}')
        fig_ax.legend()
        fig_ax.grid(True)
    
    def _plot_robustness_analysis(self):
        """Plot robustness analysis across noise levels."""
        noise_levels = ['1pct', '5pct', '10pct']
        
        data = []
        for model in self.models:
            for family in self.families:
                key = f"{model}_{family}"
                if key in self.results:
                    for noise in noise_levels:
                        metric_key = f'noise_sensitivity_{noise}'
                        data.append({
                            'Model': model,
                            'Family': family,
                            'Noise Level': noise,
                            'Sensitivity': self.results[key]['robustness'][metric_key]
                        })
        
        df = pd.DataFrame(data)
        
        # Box plot of noise sensitivity
        sns.boxplot(data=df, x='Noise Level', y='Sensitivity', hue='Model')
        plt.title('Robustness: Noise Sensitivity Analysis')
        plt.ylabel('Sensitivity Score')
        plt.yscale('log')
    
    def _plot_training_time_vs_accuracy(self):
        """Plot training time vs accuracy trade-off."""
        training_times = []
        accuracies = []
        labels = []
        colors = []
        
        color_map = {'stacked_history': 'red', 'method_of_steps': 'blue', 'memory_kernel': 'green'}
        
        for model in self.models:
            for family in self.families:
                key = f"{model}_{family}"
                if key in self.results:
                    training_times.append(self.results[key]['training_time'])
                    accuracies.append(1 - self.results[key]['accuracy']['rel_l2_error'])  # Higher is better
                    labels.append(f"{model}_{family}")
                    colors.append(color_map[model])
        
        plt.scatter(training_times, accuracies, c=colors, s=100, alpha=0.7)
        
        # Add model legend
        for model, color in color_map.items():
            plt.scatter([], [], c=color, label=model, s=100)
        
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Accuracy (1 - Rel L2 Error)')
        plt.title('Training Time vs Accuracy Trade-off')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_params_vs_performance(self):
        """Plot parameter count vs overall performance."""
        param_counts = []
        performance_scores = []
        labels = []
        
        for model in self.models:
            for family in self.families:
                key = f"{model}_{family}"
                if key in self.results:
                    param_counts.append(self.results[key]['efficiency']['param_count'])
                    
                    # Composite performance score
                    acc = 1 - min(self.results[key]['accuracy']['rel_l2_error'], 1.0)
                    corr = self.results[key]['accuracy']['correlation']
                    performance = (acc + corr) / 2
                    performance_scores.append(performance)
                    
                    labels.append(f"{model}_{family}")
        
        plt.scatter(param_counts, performance_scores, s=100, alpha=0.7)
        
        for i, label in enumerate(labels):
            plt.annotate(label, (param_counts[i], performance_scores[i]), 
                        fontsize=8, ha='center', va='bottom')
        
        plt.xlabel('Parameter Count')
        plt.ylabel('Performance Score')
        plt.title('Model Complexity vs Performance')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
    
    def _plot_temporal_dynamics(self):
        """Plot temporal dynamics quality metrics."""
        metrics = ['frequency_correlation', 'spectral_energy_ratio', 'autocorr_similarity']
        
        data = []
        for model in self.models:
            for family in self.families:
                key = f"{model}_{family}"
                if key in self.results:
                    for metric in metrics:
                        data.append({
                            'Model': model,
                            'Family': family,
                            'Metric': metric,
                            'Value': self.results[key]['temporal_dynamics'][metric]
                        })
        
        df = pd.DataFrame(data)
        
        # Grouped bar plot
        pivot_df = df.pivot_table(values='Value', index='Model', columns='Metric')
        pivot_df.plot(kind='bar', ax=plt.gca())
        plt.title('Temporal Dynamics Quality')
        plt.ylabel('Quality Score')
        plt.xlabel('Model Type')
        plt.xticks(rotation=45)
        plt.legend(title='Temporal Metrics')
    
    def _plot_cross_dataset_performance(self):
        """Plot cross-dataset performance comparison."""
        performance_matrix = np.zeros((len(self.models), len(self.families)))
        
        for i, model in enumerate(self.models):
            for j, family in enumerate(self.families):
                key = f"{model}_{family}"
                if key in self.results:
                    # Composite performance score
                    acc = 1 - min(self.results[key]['accuracy']['rel_l2_error'], 1.0)
                    corr = max(0, self.results[key]['accuracy']['correlation'])
                    performance_matrix[i, j] = (acc + corr) / 2
                else:
                    performance_matrix[i, j] = 0
        
        sns.heatmap(performance_matrix,
                   xticklabels=self.families,
                   yticklabels=self.models,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlBu',
                   cbar_kws={'label': 'Performance Score'})
        plt.title('Cross-Dataset Performance')
        plt.xlabel('Dataset Family')
        plt.ylabel('Model Type')
    
    def _plot_model_ranking(self):
        """Plot overall model ranking across all metrics."""
        model_scores = {model: [] for model in self.models}
        
        for model in self.models:
            for family in self.families:
                key = f"{model}_{family}"
                if key in self.results:
                    # Normalize and combine multiple metrics
                    acc = 1 - min(self.results[key]['accuracy']['rel_l2_error'], 1.0)
                    corr = max(0, self.results[key]['accuracy']['correlation'])
                    stab = max(0, 1 - abs(self.results[key]['stability']['lyapunov_estimate']))
                    rob = 1 - min(self.results[key]['robustness']['avg_noise_sensitivity'], 1.0)
                    
                    overall_score = (acc + corr + stab + rob) / 4
                    model_scores[model].append(overall_score)
        
        # Calculate mean and std for each model
        model_means = {model: np.mean(scores) for model, scores in model_scores.items()}
        model_stds = {model: np.std(scores) for model, scores in model_scores.items()}
        
        models = list(model_means.keys())
        means = list(model_means.values())
        stds = list(model_stds.values())
        
        plt.bar(models, means, yerr=stds, capsize=5, alpha=0.7)
        plt.title('Overall Model Ranking')
        plt.ylabel('Overall Performance Score')
        plt.xlabel('Model Type')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for i, (model, mean) in enumerate(zip(models, means)):
            plt.text(i, mean + stds[i] + 0.01, f'{mean:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
    
    def generate_summary_report(self):
        """Generate a comprehensive text summary report."""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE DNO EVALUATION SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        total_combinations = len(self.results)
        report.append(f"Total Model-Family Combinations Evaluated: {total_combinations}")
        report.append("")
        
        # Best performing combinations
        report.append("🏆 TOP PERFORMERS BY CATEGORY:")
        report.append("")
        
        # Best accuracy
        best_acc = min(self.results.items(), 
                      key=lambda x: x[1]['accuracy']['rel_l2_error'])
        report.append(f"Best Accuracy: {best_acc[0]} (Rel L2 Error: {best_acc[1]['accuracy']['rel_l2_error']:.6f})")
        
        # Best stability
        best_stab = min(self.results.items(),
                       key=lambda x: abs(x[1]['stability']['lyapunov_estimate']))
        report.append(f"Best Stability: {best_stab[0]} (Lyapunov: {best_stab[1]['stability']['lyapunov_estimate']:.6f})")
        
        # Best efficiency
        best_eff = min(self.results.items(),
                      key=lambda x: x[1]['efficiency']['inference_time_ms'])
        report.append(f"Best Efficiency: {best_eff[0]} (Inference: {best_eff[1]['efficiency']['inference_time_ms']:.2f}ms)")
        
        # Best robustness
        best_rob = min(self.results.items(),
                      key=lambda x: x[1]['robustness']['avg_noise_sensitivity'])
        report.append(f"Best Robustness: {best_rob[0]} (Noise Sensitivity: {best_rob[1]['robustness']['avg_noise_sensitivity']:.6f})")
        
        report.append("")
        report.append("📊 MODEL COMPARISON:")
        report.append("")
        
        # Model-wise analysis
        for model in self.models:
            model_results = [v for k, v in self.results.items() if k.startswith(model)]
            if model_results:
                avg_acc = np.mean([r['accuracy']['rel_l2_error'] for r in model_results])
                avg_time = np.mean([r['training_time'] for r in model_results])
                avg_params = np.mean([r['efficiency']['param_count'] for r in model_results])
                
                report.append(f"{model.upper()}:")
                report.append(f"  • Average Rel L2 Error: {avg_acc:.6f}")
                report.append(f"  • Average Training Time: {avg_time:.2f}s")
                report.append(f"  • Average Parameters: {avg_params:,.0f}")
                report.append("")
        
        # Dataset-wise analysis
        report.append("📈 DATASET ANALYSIS:")
        report.append("")
        
        for family in self.families:
            family_results = [v for k, v in self.results.items() if k.endswith(family)]
            if family_results:
                avg_acc = np.mean([r['accuracy']['rel_l2_error'] for r in family_results])
                best_model = min(family_results, key=lambda x: x['accuracy']['rel_l2_error'])
                
                report.append(f"{family.upper()}:")
                report.append(f"  • Average Rel L2 Error: {avg_acc:.6f}")
                report.append(f"  • Best Model: {best_model['model_type']} ({best_model['accuracy']['rel_l2_error']:.6f})")
                report.append("")
        
        # Save report
        report_text = "\n".join(report)
        with open('evaluation_plots/summary_report.txt', 'w') as f:
            f.write(report_text)
        
        print("📄 Summary report saved to evaluation_plots/summary_report.txt")
        print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description="Visualize DNO Evaluation Results")
    parser.add_argument('--results', type=str, required=True,
                       help='Path to evaluation results JSON file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"Error: Results file {args.results} not found!")
        return
    
    visualizer = EvaluationVisualizer(args.results)
    
    print("Creating comprehensive visualization dashboard...")
    visualizer.create_comprehensive_dashboard()
    
    print("Generating summary report...")
    visualizer.generate_summary_report()
    
    print("\n🎉 Visualization and analysis complete!")
    print("Check the 'evaluation_plots' directory for all generated files.")


if __name__ == "__main__":
    main()
