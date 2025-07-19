#!/usr/bin/env python3
"""
test_comprehensive_evaluation.py
================================
Test script to verify comprehensive evaluation works for all 12 model-family combinations.
This validates the complete DNO evaluation pipeline end-to-end.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_evaluation(model, family, epochs=1):
    """Run comprehensive evaluation for a specific model-family combination."""
    print(f"\n{'='*60}")
    print(f"Testing: {model.upper()} on {family.upper()}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "comprehensive_evaluation_fixed.py",
        "--model", model,
        "--family", family,
        "--epochs", str(epochs)
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS ({duration:.1f}s)")
            # Extract key metrics from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'COMPREHENSIVE EVALUATION SUMMARY' in line:
                    break
            for line in lines:
                if any(metric in line for metric in ['MAE:', 'Parameters:', 'Inference Time:']):
                    print(f"   {line.strip()}")
            return True
        else:
            print(f"❌ FAILED ({duration:.1f}s)")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT (>300s)")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return False

def main():
    """Test all 12 model-family combinations."""
    models = ["stacked_history", "method_of_steps", "memory_kernel"]
    families = ["mackey", "delayed_logistic", "neutral", "reaction_diffusion"]
    
    print("🚀 COMPREHENSIVE EVALUATION PIPELINE TEST")
    print("Testing all 12 model-family combinations...")
    
    results = {}
    total_tests = len(models) * len(families)
    passed_tests = 0
    
    for model in models:
        for family in families:
            success = run_evaluation(model, family, epochs=1)
            results[f"{model}_{family}"] = success
            if success:
                passed_tests += 1
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    print(f"\nDetailed Results:")
    for combo, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {combo:<35} {status}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED! Comprehensive evaluation pipeline is fully functional.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
