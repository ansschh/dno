#!/bin/bash
# run_full_hyperparameter_search.sh
# Complete hyperparameter search across all models and datasets
# This will run 12 comprehensive searches (3 models × 4 datasets)

echo "🚀 Starting Complete DNO Hyperparameter Search"
echo "=============================================="
echo "This will run 12 comprehensive searches:"
echo "- 3 models: stacked_history, method_of_steps, memory_kernel"
echo "- 4 datasets: mackey, delayed_logistic, neutral, reaction_diffusion"
echo "- 100 trials each with up to 200 epochs"
echo ""

# Models to test
MODELS=("stacked_history" "method_of_steps" "memory_kernel")

# Datasets to test
FAMILIES=("mackey" "delayed_logistic" "neutral" "reaction_diffusion")

# Search parameters
N_TRIALS=100
MAX_EPOCHS=200
SEARCH_TYPE="bayesian"

echo "Starting searches at $(date)"
echo ""

# Counter for progress tracking
TOTAL_SEARCHES=$((${#MODELS[@]} * ${#FAMILIES[@]}))
CURRENT_SEARCH=0

# Run all combinations
for MODEL in "${MODELS[@]}"; do
    for FAMILY in "${FAMILIES[@]}"; do
        CURRENT_SEARCH=$((CURRENT_SEARCH + 1))
        
        echo "[$CURRENT_SEARCH/$TOTAL_SEARCHES] Running: $MODEL on $FAMILY dataset"
        echo "Command: python hyperparameter_search.py --search_type $SEARCH_TYPE --model $MODEL --family $FAMILY --n_trials $N_TRIALS --max_epochs $MAX_EPOCHS"
        
        # Run the hyperparameter search
        python hyperparameter_search.py \
            --search_type $SEARCH_TYPE \
            --model $MODEL \
            --family $FAMILY \
            --n_trials $N_TRIALS \
            --max_epochs $MAX_EPOCHS \
            --wandb_project "dno-comprehensive-search"
        
        if [ $? -eq 0 ]; then
            echo "✅ Completed: $MODEL on $FAMILY"
        else
            echo "❌ Failed: $MODEL on $FAMILY"
        fi
        
        echo "----------------------------------------"
        echo ""
    done
done

echo "🎉 All hyperparameter searches completed!"
echo "Finished at $(date)"
echo ""
echo "Next steps:"
echo "1. Generate comprehensive visualizations:"
echo "   python advanced_visualizations.py"
echo "2. Check results in hyperparameter_results/ directory"
echo "3. Review W&B dashboard for detailed analysis"
