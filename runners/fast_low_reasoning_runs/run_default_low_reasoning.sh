#!/bin/bash
# run_experiments.sh - Run BO experiments sequentially with venv activation
# Usage - Copy to batched run folder, adjust the "Experiment Folder" variable, chmod +x the file, then run with ./{name}.sh
  
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

# Define the folder containing experiment scripts
EXPERIMENT_FOLDER="./fast_low_reasoning_runs"

# Check if folder exists
if [ ! -d "$EXPERIMENT_FOLDER" ]; then
    echo "ERROR: Experiment folder not found at $EXPERIMENT_FOLDER"#!/bin/bash
# run_experiments.sh - Run BO experiments sequentially with venv activation
  
# Set up PID-based log file
LOG_FILE="log.$$"
exec > >(tee -a "$LOG_FILE") 2>&1
  
echo "=========================================="
echo "Starting Batch BO Experiments"
echo "Started at: $(date)"
echo "Logging to: $LOG_FILE (PID: $$)"
echo "=========================================="
  
# Get the directory where this script is located (the batch folder)
BATCH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Batch directory: $BATCH_DIR"

# Navigate to project root (two levels up from the batch folder)
PROJECT_ROOT="$( cd "$BATCH_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"
echo "Project root: $(pwd)"
  
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
    local basename="$(basename "$script")"
    echo ""
    echo "=========================================="
    echo "Starting: $basename"
    echo "Script:   $script"
    echo "Started at: $(date)"
    echo "=========================================="
    python "$script"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: $basename failed with exit code $exit_code"
    else
        echo "✓ $basename completed successfully"
    fi
    echo "Finished at: $(date)"
    return $exit_code
}

# Count Python files in the batch directory
PYTHON_FILES=("$BATCH_DIR"/*.py)
FILE_COUNT=${#PYTHON_FILES[@]}

if [ ! -e "${PYTHON_FILES[0]}" ]; then
    echo "ERROR: No Python files found in $BATCH_DIR"
    exit 1
fi

echo ""
echo "Found $FILE_COUNT Python file(s) to run from $(basename "$BATCH_DIR")"
echo ""

# Run all experiments and track results
declare -a EXIT_CODES
declare -a SCRIPT_NAMES

for script in "$BATCH_DIR"/*.py; do
    if [ -f "$script" ]; then
        run_experiment "$script"
        EXIT_CODES+=($?)
        SCRIPT_NAMES+=("$(basename "$script")")
    fi
done
  
# Deactivate virtual environment
deactivate
  
echo ""
echo "=========================================="
echo "All experiments completed at: $(date)"
echo "=========================================="
echo "Summary:"
for i in "${!SCRIPT_NAMES[@]}"; do
    printf "  %-40s exit %d\n" "${SCRIPT_NAMES[$i]}:" "${EXIT_CODES[$i]}"
done
echo "=========================================="
  
# Exit with error if any failed
FAILED=0
for code in "${EXIT_CODES[@]}"; do
    if [ "$code" -ne 0 ]; then
        FAILED=1
        break
    fi
done

if [ $FAILED -eq 1 ]; then
    echo "⚠️  One or more experiments failed"
    exit 1
fi
  
echo "✓ All experiments completed successfully"
exit 0
    exit 1
fi

echo "Experiment folder: $EXPERIMENT_FOLDER"
  
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
    local basename="$(basename "$script")"
    echo ""
    echo "=========================================="
    echo "Starting: $basename"
    echo "Script:   $script"
    echo "Started at: $(date)"
    echo "=========================================="
    python "$script"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: $basename failed with exit code $exit_code"
    else
        echo "✓ $basename completed successfully"
    fi
    echo "Finished at: $(date)"
    return $exit_code
}

# Count Python files
PYTHON_FILES=("$EXPERIMENT_FOLDER"/*.py)
FILE_COUNT=${#PYTHON_FILES[@]}

if [ ! -e "${PYTHON_FILES[0]}" ]; then
    echo "ERROR: No Python files found in $EXPERIMENT_FOLDER"
    exit 1
fi

echo ""
echo "Found $FILE_COUNT Python file(s) to run"
echo ""

# Run all experiments and track results
declare -a EXIT_CODES
declare -a SCRIPT_NAMES

for script in "$EXPERIMENT_FOLDER"/*.py; do
    if [ -f "$script" ]; then
        run_experiment "$script"
        EXIT_CODES+=($?)
        SCRIPT_NAMES+=("$(basename "$script")")
    fi
done
  
# Deactivate virtual environment
deactivate
  
echo ""
echo "=========================================="
echo "All experiments completed at: $(date)"
echo "=========================================="
echo "Summary:"
for i in "${!SCRIPT_NAMES[@]}"; do
    printf "  %-40s exit %d\n" "${SCRIPT_NAMES[$i]}:" "${EXIT_CODES[$i]}"
done
echo "=========================================="
  
# Exit with error if any failed
FAILED=0
for code in "${EXIT_CODES[@]}"; do
    if [ "$code" -ne 0 ]; then
        FAILED=1
        break
    fi
done

if [ $FAILED -eq 1 ]; then
    echo "⚠️  One or more experiments failed"
    exit 1
fi
  
echo "✓ All experiments completed successfully"
exit 0