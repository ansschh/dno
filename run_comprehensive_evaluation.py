"""
run_comprehensive_evaluation.py
===============================
Master script to run comprehensive DNO evaluation and generate all visualizations.
This script orchestrates the entire evaluation pipeline from training to visualization.

Usage:
    python run_comprehensive_evaluation.py --quick    # Quick test (1 epoch each)
    python run_comprehensive_evaluation.py --full     # Full evaluation (10 epochs each)
    python run_comprehensive_evaluation.py --custom --epochs 5  # Custom epochs
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors gracefully."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print(f"❌ FAILED: {description}")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {description}")
        print(f"Error: {e}")
        return False
    
    elapsed = time.time() - start_time
    print(f"⏱️ Completed in {elapsed:.2f} seconds")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run Comprehensive DNO Evaluation Pipeline")
    parser.add_argument('--quick', action='store_true', help='Quick evaluation (1 epoch each)')
    parser.add_argument('--full', action='store_true', help='Full evaluation (10 epochs each)')
    parser.add_argument('--custom', action='store_true', help='Custom evaluation')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for custom evaluation')
    parser.add_argument('--data_root', type=str, default='data/combined', help='Data root directory')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, only run visualization')
    
    args = parser.parse_args()
    
    if not (args.quick or args.full or args.custom):
        print("Error: Must specify --quick, --full, or --custom")
        return
    
    # Determine epochs
    if args.quick:
        epochs = 1
        eval_type = "quick"
    elif args.full:
        epochs = 10
        eval_type = "full"
    else:
        epochs = args.epochs
        eval_type = "custom"
    
    print(f"""
    🔬 COMPREHENSIVE DNO EVALUATION PIPELINE
    ========================================
    
    Evaluation Type: {eval_type.upper()}
    Epochs per model: {epochs}
    Data Root: {args.data_root}
    Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    This pipeline will:
    1. Train and evaluate all 12 model-family combinations
    2. Compute comprehensive metrics (accuracy, stability, generalization, etc.)
    3. Generate publication-ready visualizations
    4. Create summary reports
    
    """)
    
    # Check if data directory exists
    if not os.path.exists(args.data_root):
        print(f"❌ Error: Data directory {args.data_root} not found!")
        print("Please ensure the dataset files are available.")
        return
    
    # Step 1: Run comprehensive evaluation
    if not args.skip_training:
        cmd = f"python comprehensive_evaluation.py --all --epochs {epochs} --data_root {args.data_root}"
        success = run_command(cmd, f"Comprehensive Evaluation ({epochs} epochs)")
        
        if not success:
            print("❌ Evaluation failed. Stopping pipeline.")
            return
    else:
        print("⏭️ Skipping training phase as requested.")
    
    # Step 2: Check if results file exists
    results_file = "evaluation_results/comprehensive_evaluation_all_combinations.json"
    if not os.path.exists(results_file):
        print(f"❌ Error: Results file {results_file} not found!")
        print("Cannot proceed with visualization without evaluation results.")
        return
    
    # Step 3: Generate visualizations
    cmd = f"python evaluation_visualizer.py --results {results_file}"
    success = run_command(cmd, "Generating Comprehensive Visualizations")
    
    if not success:
        print("❌ Visualization failed.")
        return
    
    # Step 4: Create additional analysis (if needed)
    print(f"\n{'='*60}")
    print("📊 GENERATING ADDITIONAL ANALYSIS")
    print(f"{'='*60}")
    
    # Create a simple performance comparison table
    try:
        import json
        import pandas as pd
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create performance summary table
        summary_data = []
        for key, result in results.items():
            model_type, family = key.split('_', 1)
            summary_data.append({
                'Model': model_type,
                'Dataset': family,
                'Rel_L2_Error': result['accuracy']['rel_l2_error'],
                'Correlation': result['accuracy']['correlation'],
                'Training_Time_s': result['training_time'],
                'Parameters': result['efficiency']['param_count'],
                'Inference_ms': result['efficiency']['inference_time_ms'],
                'Noise_Sensitivity': result['robustness']['avg_noise_sensitivity']
            })
        
        df = pd.DataFrame(summary_data)
        
        # Save to CSV
        os.makedirs('evaluation_results', exist_ok=True)
        csv_file = 'evaluation_results/performance_summary.csv'
        df.to_csv(csv_file, index=False)
        print(f"📈 Performance summary saved to {csv_file}")
        
        # Print top performers
        print("\n🏆 TOP PERFORMERS:")
        print("\nBest Accuracy (Lowest Rel L2 Error):")
        best_acc = df.loc[df['Rel_L2_Error'].idxmin()]
        print(f"  {best_acc['Model']} on {best_acc['Dataset']}: {best_acc['Rel_L2_Error']:.6f}")
        
        print("\nBest Correlation:")
        best_corr = df.loc[df['Correlation'].idxmax()]
        print(f"  {best_corr['Model']} on {best_corr['Dataset']}: {best_corr['Correlation']:.4f}")
        
        print("\nFastest Training:")
        fastest = df.loc[df['Training_Time_s'].idxmin()]
        print(f"  {fastest['Model']} on {fastest['Dataset']}: {fastest['Training_Time_s']:.2f}s")
        
        print("\nMost Robust (Lowest Noise Sensitivity):")
        most_robust = df.loc[df['Noise_Sensitivity'].idxmin()]
        print(f"  {most_robust['Model']} on {most_robust['Dataset']}: {most_robust['Noise_Sensitivity']:.6f}")
        
    except Exception as e:
        print(f"⚠️ Could not generate additional analysis: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎉 COMPREHENSIVE EVALUATION COMPLETE!")
    print(f"{'='*60}")
    print(f"""
    ✅ All 12 model-family combinations evaluated
    ✅ Comprehensive metrics computed
    ✅ Visualizations generated
    ✅ Summary reports created
    
    📁 Generated Files:
    • evaluation_results/comprehensive_evaluation_all_combinations.json
    • evaluation_results/performance_summary.csv
    • evaluation_plots/comprehensive_dashboard.png
    • evaluation_plots/comprehensive_dashboard.pdf
    • evaluation_plots/summary_report.txt
    
    🔬 Evaluation completed in {eval_type} mode with {epochs} epochs per model.
    
    Next steps:
    1. Review the comprehensive dashboard (evaluation_plots/comprehensive_dashboard.png)
    2. Read the detailed summary report (evaluation_plots/summary_report.txt)
    3. Analyze the performance CSV for detailed comparisons
    4. Use these results for your research publication!
    """)

if __name__ == "__main__":
    main()
