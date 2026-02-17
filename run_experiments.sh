#!/bin/bash

# run_experiments.sh - Run BO experiments sequentially with venv activation

echo "=========================================="
echo "Starting Batch BO Experiments"
echo "Started at: $(date)"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Script directory: $SCRIPT_DIR"

# Navigate to project root (where the script is)
cd "$SCRIPT_DIR"
echo "Working directory: $(pwd)"

# Activate virtual environment
VENV_PATH="./.venv"  # Updated to .venv

if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    echo "Python location: $(which python)"
    echo "Python version: $(python --version)"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    echo "Please create a venv with: python3 -m venv .venv"
    exit 1
fi

# Verify required packages (optional but recommended)
echo ""
echo "Verifying Python environment..."
python -c "import torch; import botorch; print('✓ PyTorch and BoTorch installed')" || {
    echo "ERROR: Required packages not installed"
    echo "Please run: pip install torch botorch numpy pandas matplotlib scipy python-dotenv"
    exit 1
}

# Run high statistics experiment first
echo ""
echo "=========================================="
echo "Starting high statistics experiment (50 seeds, beta=2.0)..."
echo "Started at: $(date)"
echo "=========================================="

python run_high_statistics.py
HIGH_STATS_EXIT=$?

if [ $HIGH_STATS_EXIT -ne 0 ]; then
    echo "ERROR: High statistics experiment failed with exit code $HIGH_STATS_EXIT"
    echo "Continuing to grid search anyway..."
else
    echo "✓ High statistics experiment completed successfully"
fi

echo ""
echo "High statistics experiment completed at: $(date)"
echo ""

# Run beta grid search experiment
echo "=========================================="
echo "Starting beta grid search experiment (10 seeds, 5 betas)..."
echo "Started at: $(date)"
echo "=========================================="

python run_beta_grid_search.py
GRID_EXIT=$?

if [ $GRID_EXIT -ne 0 ]; then
    echo "ERROR: Grid search experiment failed with exit code $GRID_EXIT"
else
    echo "✓ Grid search experiment completed successfully"
fi

echo ""
echo "=========================================="
echo "All experiments completed at: $(date)"
echo "=========================================="
echo "Summary:"
echo "  High statistics exit code: $HIGH_STATS_EXIT"
echo "  Grid search exit code: $GRID_EXIT"
echo "=========================================="

# Deactivate virtual environment
deactivate

# Exit with error if either failed
if [ $HIGH_STATS_EXIT -ne 0 ] || [ $GRID_EXIT -ne 0 ]; then
    echo "⚠️  One or more experiments failed"
    exit 1
fi

echo "✓ All experiments completed successfully"
exit 0