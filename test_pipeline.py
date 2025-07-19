"""
test_pipeline.py
================
Quick test script to demonstrate the DNO hyperparameter search pipeline.
Runs a minimal search to validate the complete workflow.
"""

import os
import sys
import subprocess
from pathlib import Path

def test_pipeline():
    """Run a minimal test of the complete pipeline."""
    print("🚀 Testing DNO Hyperparameter Search Pipeline")
    print("=" * 50)
    
    # Check if data exists
    data_dir = Path("./data/combined")
    if not data_dir.exists():
        print("❌ Data directory not found. Please ensure datasets are available.")
        print("   Expected: ./data/combined/{family}_train.pkl and {family}_test.pkl")
        return False
    
    # List available datasets
    train_files = list(data_dir.glob("*_train.pkl"))
    if not train_files:
        print("❌ No training datasets found in ./data/combined/")
        return False
    
    print(f"✅ Found {len(train_files)} training datasets:")
    for f in train_files:
        print(f"   - {f.name}")
    
    # Run a quick hyperparameter search test
    print("\n🔍 Running minimal hyperparameter search test...")
    
    # Use the first available dataset
    family = train_files[0].stem.replace("_train", "")
    
    cmd = [
        sys.executable, "hyperparameter_search.py",
        "--search_type", "random",
        "--model", "method_of_steps", 
        "--family", family,
        "--n_trials", "2",  # Minimal for testing
        "--max_epochs", "5",  # Very short for testing
        "--wandb_project", "dno-pipeline-test"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Hyperparameter search test completed successfully!")
            print("\n📊 Testing visualization generation...")
            
            # Test visualization generation
            vis_cmd = [sys.executable, "advanced_visualizations.py"]
            vis_result = subprocess.run(vis_cmd, capture_output=True, text=True, timeout=120)
            
            if vis_result.returncode == 0:
                print("✅ Visualization generation completed successfully!")
                print("\n🎉 Pipeline test PASSED!")
                
                # Show generated files
                results_dir = Path("hyperparameter_results")
                vis_dir = Path("visualizations")
                
                if results_dir.exists():
                    result_files = list(results_dir.rglob("*"))
                    print(f"\n📁 Generated {len(result_files)} result files in hyperparameter_results/")
                
                if vis_dir.exists():
                    vis_files = list(vis_dir.glob("*"))
                    print(f"📁 Generated {len(vis_files)} visualization files in visualizations/")
                    for f in vis_files:
                        print(f"   - {f.name}")
                
                return True
            else:
                print("❌ Visualization generation failed:")
                print(vis_result.stderr)
                return False
        else:
            print("❌ Hyperparameter search test failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def main():
    """Main test function."""
    print("DNO Pipeline Test")
    print("================")
    print("This script tests the complete hyperparameter search and visualization pipeline.")
    print("It will run a minimal search with 2 trials and 5 epochs for validation.\n")
    
    success = test_pipeline()
    
    if success:
        print("\n✅ PIPELINE TEST SUCCESSFUL!")
        print("\nNext steps:")
        print("1. Run full hyperparameter search:")
        print("   python hyperparameter_search.py --search_type bayesian --n_trials 100 --max_epochs 200")
        print("2. Generate comprehensive visualizations:")
        print("   python advanced_visualizations.py")
        print("3. Check results in hyperparameter_results/ and visualizations/")
    else:
        print("\n❌ PIPELINE TEST FAILED!")
        print("Please check the error messages above and ensure:")
        print("1. All dependencies are installed (pip install -r requirements.txt)")
        print("2. Dataset files are available in ./data/combined/")
        print("3. CUDA is available if using GPU training")

if __name__ == "__main__":
    main()
