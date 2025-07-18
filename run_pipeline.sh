#!/bin/bash
# ======================================================
# DNO Pipeline Script
# ======================================================
# This script runs the full pipeline for training and evaluating
# the three Delay Neural Operator variants on all dataset families.
# 
# Usage:
#   bash run_pipeline.sh [--data_dir /path/to/data] [--epochs N] [--quick_test]
#
# Options:
#   --data_dir     Path to data directory (default: ./data)
#   --epochs       Number of epochs for training (default: 50)
#   --quick_test   Run a quick test with fewer epochs (default: false)
#   --skip_train   Skip training and only run evaluation (default: false)

# Default paths
DATA_DIR="./data"
EPOCHS=50
BATCH_SIZE=64
QUICK_TEST=0
SKIP_TRAIN=0
CHECKPOINT_DIR="./checkpoints"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --quick_test)
      QUICK_TEST=1
      EPOCHS=3
      BATCH_SIZE=16
      shift
      ;;
    --skip_train)
      SKIP_TRAIN=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create directories if they don't exist
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "./results"

# Dataset families with their file names
declare -A FILE_NAMES
FAMILIES=("mackey_glass" "delayed_logistic" "neutral_dde" "reaction_diffusion")
FILE_NAMES["mackey_glass"]="mackey_glass"
FILE_NAMES["delayed_logistic"]="delayed_logistic"
FILE_NAMES["neutral_dde"]="neutral_dde"
FILE_NAMES["reaction_diffusion"]="reaction_diffusion"

# Color formatting for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}         Delay Neural Operator (DNO) Pipeline         ${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "Data directory: ${YELLOW}$DATA_DIR${NC}"
echo -e "Checkpoint directory: ${YELLOW}$CHECKPOINT_DIR${NC}"
echo -e "Epochs: ${YELLOW}$EPOCHS${NC}"
echo -e "Quick test mode: ${YELLOW}$([ $QUICK_TEST -eq 1 ] && echo "Yes" || echo "No")${NC}"

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
  echo -e "${RED}ERROR: Data directory $DATA_DIR does not exist!${NC}"
  exit 1
fi

# Debug: List all files in the data directory
echo -e "${YELLOW}Listing files in $DATA_DIR:${NC}"
ls -la "$DATA_DIR"

# Create combined directory if it doesn't exist
mkdir -p "$DATA_DIR/combined"

# Check if we need to create dataset files for the train/test splits
for family in "${FAMILIES[@]}"; do
  # Convert family name to the format expected by the scripts
  script_family="$family"
  if [ "$family" == "mackey_glass" ]; then
    script_family="mackey"
  elif [ "$family" == "neutral_dde" ]; then
    script_family="neutral"
  fi
  
  # Get the actual file name
  file_name=${FILE_NAMES[$family]}
  
  echo -e "${YELLOW}Processing $family (script_family=$script_family, file_name=$file_name)${NC}"
  
  # Check if train/test splits exist with the expected script names
  if [ ! -f "$DATA_DIR/combined/${script_family}_train.pkl" ] || [ ! -f "$DATA_DIR/combined/${script_family}_test.pkl" ]; then
    echo -e "${YELLOW}Creating train/test split files for $family${NC}"
    
    # First check if files exist in the combined directory with the family name
    if [ -f "$DATA_DIR/combined/${file_name}_train.pkl" ]; then
      echo -e "${GREEN}Creating symlink from $DATA_DIR/combined/${file_name}_train.pkl to $DATA_DIR/combined/${script_family}_train.pkl${NC}"
      # Use absolute paths for symlinks
      TRAIN_ABS_PATH=$(readlink -f "$DATA_DIR/combined/${file_name}_train.pkl")
      ln -sf "$TRAIN_ABS_PATH" "$DATA_DIR/combined/${script_family}_train.pkl"
    else
      echo -e "${RED}Error: Could not find ${file_name}_train.pkl in combined directory${NC}"
      # Try to find in the family directory
      if [ -f "$DATA_DIR/$family/$family.pkl" ]; then
        echo -e "${GREEN}Copying $DATA_DIR/$family/$family.pkl to $DATA_DIR/combined/${script_family}_train.pkl${NC}"
        cp "$DATA_DIR/$family/$family.pkl" "$DATA_DIR/combined/${script_family}_train.pkl"
      elif [ -f "$DATA_DIR/$family.pkl" ]; then
        echo -e "${GREEN}Copying $DATA_DIR/$family.pkl to $DATA_DIR/combined/${script_family}_train.pkl${NC}"
        cp "$DATA_DIR/$family.pkl" "$DATA_DIR/combined/${script_family}_train.pkl"
      else
        echo -e "${RED}Error: Could not find any suitable train file for $family${NC}"
        ls -la "$DATA_DIR/combined" | grep -i "${file_name}"
      fi
    fi
    
    if [ -f "$DATA_DIR/combined/${file_name}_test.pkl" ]; then
      echo -e "${GREEN}Creating symlink from $DATA_DIR/combined/${file_name}_test.pkl to $DATA_DIR/combined/${script_family}_test.pkl${NC}"
      # Use absolute paths for symlinks
      TEST_ABS_PATH=$(readlink -f "$DATA_DIR/combined/${file_name}_test.pkl")
      ln -sf "$TEST_ABS_PATH" "$DATA_DIR/combined/${script_family}_test.pkl"
    else
      echo -e "${RED}Error: Could not find ${file_name}_test.pkl in combined directory${NC}"
      # Try to find in the family directory
      if [ -f "$DATA_DIR/$family/$family.pkl" ]; then
        echo -e "${GREEN}Copying $DATA_DIR/$family/$family.pkl to $DATA_DIR/combined/${script_family}_test.pkl${NC}"
        cp "$DATA_DIR/$family/$family.pkl" "$DATA_DIR/combined/${script_family}_test.pkl"
      elif [ -f "$DATA_DIR/$family.pkl" ]; then
        echo -e "${GREEN}Copying $DATA_DIR/$family.pkl to $DATA_DIR/combined/${script_family}_test.pkl${NC}"
        cp "$DATA_DIR/$family.pkl" "$DATA_DIR/combined/${script_family}_test.pkl"
      else
        echo -e "${RED}Error: Could not find any suitable test file for $family${NC}"
        ls -la "$DATA_DIR/combined" | grep -i "${file_name}"
      fi
    fi
  fi
done

# Debug: List all files in the combined directory
echo -e "${YELLOW}Listing files in $DATA_DIR/combined:${NC}"
ls -la "$DATA_DIR/combined"

# Function to run model training for a specific variant and family
run_training() {
  local variant=$1
  local family=$2
  local script_name=""
  local extra_args=""
  
  # Determine script file and model-specific parameters
  case $variant in
    "stacked_history")
      script_name="stacked_history.py"
      # Add extra args for higher-dim problems
      if [ "$family" == "reaction_diffusion" ]; then
        extra_args="--fourier_modes_s 6 --fourier_modes_x 16"
      fi
      ;;
    "method_of_steps")
      script_name="method_of_steps.py"
      # Specific args for method of steps
      if [ "$family" == "reaction_diffusion" ]; then
        extra_args="--S 64 --step_out 64 --roll_steps 8"
      fi
      ;;
    "memory_kernel")
      script_name="memory_kernel.py"
      # Specific args for memory kernel
      if [ "$family" == "delayed_logistic" ]; then
        extra_args="--euler_dt 0.025"
      fi
      if [ "$family" == "reaction_diffusion" ]; then
        extra_args="--S 64 --euler_dt 0.01"
      fi
      ;;
  esac
  
  echo -e "${GREEN}[$variant - $family] Starting training...${NC}"
  
  # Convert family name to the format expected by the scripts
  script_family="$family"
  if [ "$family" == "mackey_glass" ]; then
    script_family="mackey"
  elif [ "$family" == "neutral_dde" ]; then
    script_family="neutral"
  fi
  
  # Execute the script with parameters
  python "$script_name" \
    --family "$script_family" \
    --data_root "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    $extra_args 2>&1 | tee "./results/${variant}_${script_family}.log"
  
  # Check if the run was successful
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}[$variant - $family] Training completed successfully!${NC}"
  else
    echo -e "${RED}[$variant - $family] Training failed!${NC}"
  fi
}

# Function to run model evaluation 
run_evaluation() {
  local variant=$1
  local checkpoint_file=""
  
  case $variant in
    "stacked_history")
      script_name="stacked_history.py"
      ;;
    "method_of_steps")
      script_name="method_of_steps.py"
      ;;
    "memory_kernel")
      script_name="memory_kernel.py"
      ;;
  esac
  
  echo -e "${BLUE}===== Evaluating $variant on all families =====${NC}"
  
  for family in "${FAMILIES[@]}"; do
    # Convert family name to the format expected by the scripts
    script_family="$family"
    if [ "$family" == "mackey_glass" ]; then
      script_family="mackey"
    elif [ "$family" == "neutral_dde" ]; then
      script_family="neutral"
    fi
    
    checkpoint_file="${CHECKPOINT_DIR}/${variant}_${script_family}.pt"
    
    if [ -f "$checkpoint_file" ]; then
      echo -e "${GREEN}[$variant] Evaluating on $family dataset...${NC}"
      # Add eval mode here
      # This would depend on how your scripts handle evaluation mode
    else
      echo -e "${YELLOW}[$variant] Checkpoint for $family not found, skipping evaluation.${NC}"
    fi
  done
}

# Main Pipeline

# 1. Train all models on all datasets (unless --skip_train)
if [ $SKIP_TRAIN -eq 0 ]; then
  echo -e "${BLUE}Starting training pipeline...${NC}"
  
  for family in "${FAMILIES[@]}"; do
    echo -e "${BLUE}===== Family: $family =====${NC}"
    
    # Train Stacked-History variant
    run_training "stacked_history" "$family"
    
    # Train Method-of-Steps variant
    run_training "method_of_steps" "$family"
    
    # Train Memory-Kernel variant
    run_training "memory_kernel" "$family"
  done
else
  echo -e "${YELLOW}Training phase skipped as requested.${NC}"
fi

# 2. Collect and summarize results
echo -e "${BLUE}===== Generating summary report =====${NC}"
echo "DNO Pipeline Results Summary" > "./results/summary.txt"
echo "Date: $(date)" >> "./results/summary.txt"
echo "---------------------------------" >> "./results/summary.txt"

# Extract best performance from logs
for variant in "stacked_history" "method_of_steps" "memory_kernel"; do
  echo "Model: $variant" >> "./results/summary.txt"
  for family in "${FAMILIES[@]}"; do
    # Convert family name to the format expected by the scripts
    script_family="$family"
    if [ "$family" == "mackey_glass" ]; then
      script_family="mackey"
    elif [ "$family" == "neutral_dde" ]; then
      script_family="neutral"
    fi
    
    log_file="./results/${variant}_${script_family}.log"
    if [ -f "$log_file" ]; then
      # Extract best validation result
      best_rel=$(grep -o "New best (rel.*)" "$log_file" | tail -1)
      if [ -n "$best_rel" ]; then
        echo "  $family: $best_rel" >> "./results/summary.txt"
      else
        echo "  $family: No validation results found" >> "./results/summary.txt"
      fi
    else
      echo "  $family: No log file found" >> "./results/summary.txt"
    fi
  done
  echo "" >> "./results/summary.txt"
done

echo -e "${GREEN}Pipeline completed! Summary saved to ./results/summary.txt${NC}"
echo -e "${BLUE}======================================================${NC}"
