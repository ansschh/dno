#!/usr/bin/env python3
"""
test_all_combinations.py
========================
Simple test script to verify all 12 model-family combinations work without errors.
Runs just 1 epoch for each combination to quickly identify any remaining issues.
"""

import os
import sys
import traceback
from model_trainer import ModelTrainer
from experiment_tracker import ExperimentTracker
from hyperparameter_search import HyperparameterSearcher

def test_single_combination(model_type, family, data_root="./data"):
    """Test a single model-family combination for 1 epoch."""
    print(f"\n{'='*60}")
    print(f"Testing {model_type} on {family}")
    print(f"{'='*60}")
    
    try:
        # Basic configuration for quick test
        config = {
            'model': model_type,
            'family': family,  # Add family name for model configuration
            'fourier_modes_x': 8,
            'fourier_modes_s': 16,
            'S': 32,
            'width': 64,
            'n_layers': 2,
            'lr': 0.001,
            'batch_size': 8,
            'dropout': 0.1,
            'weight_decay': 1e-6,
            'max_epochs': 1,  # Just 1 epoch for quick test
            'data_root': data_root
        }
        
        # Initialize trainer and searcher with tracker
        trainer = ModelTrainer()
        tracker = ExperimentTracker(project_name="test_pipeline")
        searcher = HyperparameterSearcher(tracker)
        
        # Create data loaders using searcher
        train_loader, val_loader, test_loader = searcher.load_data(family, data_root)
        
        # Run training for 1 epoch
        result = trainer.train_and_evaluate(
            model_type=model_type,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            max_epochs=1
        )
        
        print(f"✅ SUCCESS: {model_type} on {family}")
        train_loss = result.get('train_loss', 'N/A')
        val_loss = result.get('val_loss', 'N/A')
        val_rel = result.get('val_rel_error', 'N/A')
        
        train_loss_str = f"{train_loss:.6f}" if isinstance(train_loss, (int, float)) else str(train_loss)
        val_loss_str = f"{val_loss:.6f}" if isinstance(val_loss, (int, float)) else str(val_loss)
        val_rel_str = f"{val_rel:.6f}" if isinstance(val_rel, (int, float)) else str(val_rel)
        
        print(f"   Final train loss: {train_loss_str}")
        print(f"   Final val loss: {val_loss_str}")
        print(f"   Final val rel error: {val_rel_str}")
        
        return True, result
        
    except Exception as e:
        print(f"❌ FAILED: {model_type} on {family}")
        print(f"   Error: {str(e)}")
        print(f"   Traceback:")
        traceback.print_exc()
        return False, str(e)

def main():
    """Test all 12 model-family combinations."""
    
    # All model types and families
    models = ['stacked_history', 'method_of_steps', 'memory_kernel']
    families = ['mackey', 'delayed_logistic', 'neutral', 'reaction_diffusion']
    
    print("Testing all 12 model-family combinations...")
    print(f"Models: {models}")
    print(f"Families: {families}")
    
    # Track results
    results = {}
    success_count = 0
    total_count = len(models) * len(families)
    
    # Test each combination
    for model in models:
        for family in families:
            success, result = test_single_combination(model, family)
            results[(model, family)] = (success, result)
            if success:
                success_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total combinations tested: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Success rate: {success_count/total_count*100:.1f}%")
    
    # Detailed results
    print(f"\nDetailed Results:")
    print(f"{'Model':<15} {'Family':<20} {'Status':<10} {'Details'}")
    print(f"{'-'*70}")
    
    for model in models:
        for family in families:
            success, result = results[(model, family)]
            status = "✅ PASS" if success else "❌ FAIL"
            if success:
                val_loss = result.get('val_loss', 'N/A')
                if isinstance(val_loss, (int, float)):
                    details = f"Loss: {val_loss:.4f}"
                else:
                    details = f"Loss: {val_loss}"
            else:
                details = str(result)[:50] + "..." if len(str(result)) > 50 else str(result)
            print(f"{model:<15} {family:<20} {status:<10} {details}")
    
    # Exit with appropriate code
    if success_count == total_count:
        print(f"\n🎉 All combinations passed! Pipeline is ready.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total_count - success_count} combinations failed. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
