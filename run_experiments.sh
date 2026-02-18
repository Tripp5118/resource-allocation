#!/bin/bash
# run_experiments.sh - Run BO experiments sequentially with venv activation

# Set up PID-based log file
LOG_FILE="log.$$"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Starting Batch BO Experiments"
echo "Started at: $(date)"
echo "Logging to: $LOG_FILE (PID: $$)"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Script directory: $SCRIPT_DIR"

# Navigate to project root (where the script is)
cd "$SCRIPT_DIR"
echo "Working directory: $(pwd)"

# Activate virtual environment
VENV_PATH="./.venv"
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

# Verify required packages
echo ""
echo "Verifying Python environment..."
python -c "import torch; import botorch; print('✓ PyTorch and BoTorch installed')" || {
    echo "ERROR: Required packages not installed"
    echo "Please run: pip install torch botorch numpy pandas matplotlib scipy python-dotenv"
    exit 1
}

# Helper function to run an experiment
run_experiment() {
    local script="$1"
    local description="$2"
    echo ""
    echo "=========================================="
    echo "Starting: $description"
    echo "Script:   $script"
    echo "Started at: $(date)"
    echo "=========================================="
    python "$script"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: $description failed with exit code $exit_code"
    else
        echo "✓ $description completed successfully"
    fi
    echo "Finished at: $(date)"
    return $exit_code
}

# Run all experiments
run_experiment "run_high_statistics.py"                  "High statistics experiment (50 seeds, beta=2.0)"
EXIT_HIGH_STATS=$?

run_experiment "run_beta_grid_search.py"                 "Beta grid search (10 seeds, 5 betas)"
EXIT_BETA_GRID=$?

run_experiment "run_high_statistics_fixed_beta.py"       "High statistics fixed beta"
EXIT_FIXED_BETA=$?

run_experiment "run_high_statistics_low_context_window.py" "High statistics low context window"
EXIT_LOW_CTX=$?

run_experiment "run_high_stats_budget_event.py"          "High stats budget event"
EXIT_BUDGET=$?

run_experiment "run_high_stats_dual_events.py"           "High stats dual events"
EXIT_DUAL=$?

run_experiment "run_high_stats_time_event.py"            "High stats time event"
EXIT_TIME=$?

# Deactivate virtual environment
deactivate

echo ""
echo "=========================================="
echo "All experiments completed at: $(date)"
echo "=========================================="
echo "Summary:"
echo "  High statistics:            exit $EXIT_HIGH_STATS"
echo "  Beta grid search:           exit $EXIT_BETA_GRID"
echo "  Fixed beta:                 exit $EXIT_FIXED_BETA"
echo "  Low context window:         exit $EXIT_LOW_CTX"
echo "  Budget event:               exit $EXIT_BUDGET"
echo "  Dual events:                exit $EXIT_DUAL"
echo "  Time event:                 exit $EXIT_TIME"
echo "=========================================="

# Exit with error if any failed
for code in $EXIT_HIGH_STATS $EXIT_BETA_GRID $EXIT_FIXED_BETA $EXIT_LOW_CTX $EXIT_BUDGET $EXIT_DUAL $EXIT_TIME; do
    if [ "$code" -ne 0 ]; then
        echo "⚠️  One or more experiments failed"
        exit 1
    fi
done

echo "✓ All experiments completed successfully"
exit 0