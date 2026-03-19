#!/bin/bash

# File: run_all.sh

set -e  # Exit on error in critical lines

mkdir -p logs

# List of scripts to run
SCRIPTS=(
  "K-CTE-BUDGET.py"
  "K-CTE-DUAL.py"
  "K-CTE-MAIN.py"
  "K-CTE-TIME.py"
  "MP-D_BUDGET.py"
  "MP-D_DUAL.py"
  "MP-D_MAIN.py"
  "MP-D_TIME.py"
)

FAILED_SCRIPTS=()

# Activate python venv
source .venv/bin/activate

echo "========== Script run started: $(date) ==========" > logs/final.out

for script in "${SCRIPTS[@]}"; do
    echo "---------- Running $script: $(date) ----------" | tee -a logs/final.out
    python "$script" >> logs/final.out 2>&1
    if [ $? -ne 0 ]; then
        echo "!!!!! ERROR: $script failed at $(date)" | tee -a logs/final.out
        FAILED_SCRIPTS+=("$script")
    else
        echo "$script completed successfully at $(date)" | tee -a logs/final.out
    fi
done

echo "========== All runs completed: $(date) ==========" | tee -a logs/final.out

if [ ${#FAILED_SCRIPTS[@]} -ne 0 ]; then
    echo "The following scripts FAILED:" | tee -a logs/final.out
    for f in "${FAILED_SCRIPTS[@]}"; do
        echo "  $f" | tee -a logs/final.out
    done
else
    echo "All scripts completed successfully!" | tee -a logs/final.out
fi