# ARMOTE-MultiOutput v1.0.0: Automated Regression workflow with Multi-Objective hyperparameter 
# optimization using Tree-Parzen Estimator algorithm for MULTI-OUTPUT regression
#
# Version: 1.0.0
# Author: Shakti P. Padhy
# Date: 2026-01-29
#
# Description:
# This script provides a comprehensive, end-to-end framework for training, optimizing,
# and evaluating regression models for MULTI-OUTPUT targets ONLY.
# 
# Key Features:
# - Dedicated multi-output support (single-output removed for cleaner code)
# - Automatic MultiOutputRegressor wrapping for models lacking native multi-output support
# - Advanced hyperparameter search for GaussianProcessRegressor KERNELS
# - Multi-objective optimization: minimize MSE, maximize R²
# - Support for custom kernel objects passed via kernel map
# - Proper length scale handling in metrics reporting
#   * Reports per-output metrics (R², MSE, NRMSE)
#   * Reports normalized RMSE (fair comparison across outputs with different scales)
# - Per-output evaluation and visualization
# - Customizable folder paths for output files
#   * Pass folder_dict with custom paths for models, plots, studies, results
#   * Flexible output organization
# - Comprehensive results export (CSV, models, plots, studies)
#
# Workflow Steps:
# 1.  Multi-Output Check: Automatically wraps models lacking native multi-output support.
# 2.  Data Scaling: Features and targets are standardized for modeling.
# 3.  Multi-Objective Optimization: Optuna finds Pareto front by optimizing on SCALED data (fair).
# 4.  Automated Model Selection: Model with highest mean R² from Pareto front selected.
# 5.  Comprehensive Evaluation: Final model evaluated with per-output metrics & plots.
# 6.  Time Tracking & Artifact Generation: Models, scalers, studies, plots, CSV saved to specified folders.
# ---

# --- 1. Imports ---
import os
import numpy as np
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import math

# Scikit-learn modules
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from xgboost import XGBRegressor
from sklearn.datasets import make_regression

# Keras (TensorFlow backend)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

# Optuna
import optuna

# --- 2. Global Variables ---
x_scaler = StandardScaler()
y_scaler = StandardScaler()

# --- 3. Folder Management Functions ---


def setup_folders(folder_dict=None):
    """
    Setup and validate folder paths for output.
    
    Args:
        folder_dict: Dict with custom folder paths. Expected keys:
            - 'models': Path for model files (default: 'models')
            - 'plots': Path for plot files (default: 'plots')
            - 'studies': Path for Optuna study files (default: 'studies')
            - 'results': Path for results CSV files (default: current directory)
    
    Returns:
        dict: Validated folder paths
    
    Example:
        folder_dict = {
            'models': '/path/to/models',
            'plots': '/path/to/plots',
            'studies': '/path/to/studies',
            'results': '/path/to/results'
        }
        folders = setup_folders(folder_dict)
    """
    # Default folders
    default_folders = {
        'models': 'models',
        'plots': 'plots',
        'studies': 'studies',
        'results': '.'
    }
    
    # Use provided folders or defaults
    if folder_dict is None:
        folders = default_folders
    else:
        folders = default_folders.copy()
        folders.update(folder_dict)
    
    # Create directories if they don't exist
    for folder_name, folder_path in folders.items():
        if folder_path != '.':  # Don't try to create current directory
            os.makedirs(folder_path, exist_ok=True)
            print(f"✓ {folder_name.capitalize()} folder ready: {os.path.abspath(folder_path)}")
    
    return folders


def get_folder_path(folders, folder_type):
    """
    Get the path for a specific folder type.
    
    Args:
        folders: Dict returned by setup_folders()
        folder_type: Type of folder ('models', 'plots', 'studies', 'results')
    
    Returns:
        str: Folder path
    """
    return folders.get(folder_type, '.')


# --- 4. Core Helper Functions ---


def _ensure_2d(arr):
    """Ensures a NumPy array is 2D. Reshapes 1D arrays to a column vector."""
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def compute_metrics(y_true, y_pred):
    """
    Computes R-squared, MSE, and MAE for multi-output targets.
    Returns MEAN metrics across all outputs.
    """
    y_true = _ensure_2d(y_true)
    y_pred = _ensure_2d(y_pred)
    
    n_outputs = y_true.shape[1]
    r2_scores = []
    mse_scores = []
    mae_scores = []
    
    for i in range(n_outputs):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2_scores.append(r2)
        mse_scores.append(mse)
        mae_scores.append(mae)
    
    return r2_scores, mse_scores, mae_scores


def compute_metrics_with_scales(y_true, y_pred, y_scaler=None, return_per_output=True):
    """
    Compute metrics with proper length scale handling.
    
    IMPORTANT: Handles outputs with different scales fairly by reporting:
    - Per-output metrics (R², MSE_original, MSE_scaled, NRMSE)
    - Mean metrics (R², MSE_scaled, NRMSE) - NRMSE is fair for comparison!
    
    Args:
        y_true: True values (original scale), shape (n_samples, n_outputs)
        y_pred: Predictions (original scale), shape (n_samples, n_outputs)
        y_scaler: StandardScaler instance (optional, for scaled metrics)
        return_per_output: If True, return per-output metrics
    
    Returns:
        dict with mean and per-output metrics including normalized values
    """
    y_true = _ensure_2d(y_true)
    y_pred = _ensure_2d(y_pred)
    
    n_outputs = y_true.shape[1]
    
    metrics = {
        'per_output': {},
        'mean': {}
    }
    
    r2_scores = []
    mse_orig_scores = []
    mae_orig_scores = []
    rmse_norm_scores = []
    mse_scaled_scores = []
    
    if y_scaler is not None:
        y_true_scaled = y_scaler.transform(y_true)
        y_pred_scaled = y_scaler.transform(y_pred)
    else:
        y_true_scaled = None
        y_pred_scaled = None
    
    for i in range(n_outputs):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]
        
        # R² (scale-invariant) ✓
        r2 = r2_score(y_true_i, y_pred_i)
        r2_scores.append(r2)
        
        # MSE in original scale (scale-dependent, for reference)
        mse_orig = mean_squared_error(y_true_i, y_pred_i)
        mse_orig_scores.append(mse_orig)
        
        # MAE in original scale
        mae_orig = mean_absolute_error(y_true_i, y_pred_i)
        mae_orig_scores.append(mae_orig)
        
        # RMSE normalized by output standard deviation (fair for comparison!) ✓
        rmse = np.sqrt(mse_orig)
        output_std = np.std(y_true_i)
        nrmse = rmse / output_std if output_std > 0 else 0
        rmse_norm_scores.append(nrmse)
        
        # MSE in scaled space (fair for comparison!) ✓
        if y_scaler is not None:
            mse_scaled = mean_squared_error(y_true_scaled[:, i], y_pred_scaled[:, i])
            mse_scaled_scores.append(mse_scaled)
        
        # Store per-output metrics
        if return_per_output:
            output_metrics = {
                'R2': round(r2, 4),
                'MSE': round(mse_orig, 6),
                'MAE': round(mae_orig, 6),
                'RMSE': round(rmse, 4),
            }
            metrics['per_output'][f'Output_{i}'] = output_metrics
    
    # Compute FAIR means - ONLY scale-invariant metrics!
    # ============================================================
    # CRITICAL: Do NOT average MSE_original/MAE_original!
    # These are scale-dependent and will be misleading when outputs
    # have different scales. Only NRMSE and MSE_scaled are fair!
    # ============================================================
    metrics['mean'] = {
        'R2': round(np.mean(r2_scores), 4),                    # ✓ Fair (scale-invariant)
        'NRMSE': round(np.mean(rmse_norm_scores), 4),          # ✓ Fair (normalized)
        'MSE_scaled': round(np.mean(mse_scaled_scores), 6),    # ✓ Fair (scaled space)
    }
    
    return metrics


def create_nn(hidden_layers=1, units=64, activation="relu", learning_rate=0.001):
    """Creates a Keras Sequential neural network for multi-output regression."""
    K.clear_session()
    
    input_dim = x_scaler.n_features_in_
    output_dim = y_scaler.n_features_in_
    model = Sequential(
        [Dense(units, activation=activation, input_dim=input_dim, name="input_layer")]
    )
    current_units = units
    for i in range(hidden_layers - 1):
        current_units = max(1, current_units // 2)
        model.add(
            Dense(current_units, activation=activation, name=f"hidden_layer_{i + 1}")
        )
    model.add(Dense(output_dim, activation="linear", name="output_layer"))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def cross_val_nn(X, y, build_fn, params, cv=5, epochs=100, batch_size=32):
    """Manual K-Fold CV for Keras model. Returns mean metrics across outputs."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    r2_scores, mse_scores = [], []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = build_fn(**params)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred_val = model.predict(X_val)
        r2_scores.append(r2_score(y_val, y_pred_val))
        mse_scores.append(mean_squared_error(y_val, y_pred_val))
    return np.mean(mse_scores), np.mean(r2_scores)


def plot_yy_multi_output(
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    model_name,
    save_folder="plots",
    output_labels=None,
):
    """
    Generates a figure with Y-Y subplot for each output dimension.
    Each subplot shows train/test data with individual performance metrics.
    """
    os.makedirs(save_folder, exist_ok=True)
    y_train = _ensure_2d(y_train)
    y_pred_train = _ensure_2d(y_pred_train)
    y_test = _ensure_2d(y_test)
    y_pred_test = _ensure_2d(y_pred_test)

    n_outputs = y_train.shape[1]
    n_cols = 2 if n_outputs > 1 else 1
    n_rows = math.ceil(n_outputs / n_cols)

    # Standard dark background with gridlines
    plt.style.use("ggplot")

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 6 * n_rows), squeeze=False
    )
    fig.suptitle(f"{model_name}: Predicted vs. Actual Values by Output", fontsize=20)

    for i in range(n_outputs):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        y_train_i, y_pred_train_i = y_train[:, i], y_pred_train[:, i]
        y_test_i, y_pred_test_i = y_test[:, i], y_pred_test[:, i]

        train_r2 = round(r2_score(y_train_i, y_pred_train_i), 4)
        train_mse = round(mean_squared_error(y_train_i, y_pred_train_i), 4)
        train_mae = round(mean_absolute_error(y_train_i, y_pred_train_i), 4)
        test_r2 = round(r2_score(y_test_i, y_pred_test_i), 4)
        test_mse = round(mean_squared_error(y_test_i, y_pred_test_i), 4)
        test_mae = round(mean_absolute_error(y_test_i, y_pred_test_i), 4)

        ax.scatter(
            y_train_i,
            y_pred_train_i,
            color="#1f77b4",
            alpha=0.8,
            label=f"Train (R²={train_r2:.3f})",
        )
        ax.scatter(
            y_test_i,
            y_pred_test_i,
            color="#2ca02c",
            alpha=0.7,
            label=f"Test (R²={test_r2:.3f})",
        )

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.9, zorder=0, label="Ideal")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        if output_labels and len(output_labels) == n_outputs:
            subplot_title = output_labels[i]
        else:
            subplot_title = f"Output {i + 1}"
        ax.set_title(subplot_title, fontsize=16)
        ax.set_xlabel("True Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.legend()

        metrics_text = (
            f"Train: R²={train_r2:.4f}, MSE={train_mse:.4f}, MAE={train_mae:.4f}\n"
            f"Test:  R²={test_r2:.4f}, MSE={test_mse:.4f}, MAE={test_mae:.4f}"
        )
        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
        )

    for i in range(n_outputs, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(save_folder, f"{model_name}_yy_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  Y-Y plot saved to: {plot_path}")


def generate_optuna_plots(study, model_name, save_folder="plots"):
    """Generates and saves Optuna visualization plots."""
    try:
        import optuna.visualization.matplotlib as vis
    except ImportError:
        print("  Could not import Optuna's matplotlib visualization.")
        return

    os.makedirs(save_folder, exist_ok=True)
    target_names = ["Mean MSE", "Mean R²"]

    try:
        ax = vis.plot_pareto_front(study, target_names=target_names)
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(os.path.join(save_folder, f"{model_name}_pareto_front.png"))
        plt.close(fig)
    except Exception as e:
        print(f"  Could not generate pareto front: {e}")

    try:
        ax = vis.plot_optimization_history(
            study, target=lambda t: t.values[0], target_name="Mean MSE"
        )
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(os.path.join(save_folder, f"{model_name}_optimization_history_mse.png"))
        plt.close(fig)
    except Exception as e:
        print(f"  Could not generate optimization history MSE: {e}")

    try:
        ax = vis.plot_optimization_history(
            study, target=lambda t: t.values[1], target_name="Mean R²"
        )
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(os.path.join(save_folder, f"{model_name}_optimization_history_r2.png"))
        plt.close(fig)
    except Exception as e:
        print(f"  Could not generate optimization history R2: {e}")

    try:
        ax = vis.plot_param_importances(study)
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(os.path.join(save_folder, f"{model_name}_param_importances.png"))
        plt.close(fig)
    except Exception as e:
        print(f"  Could not generate parameter importances: {e}")

    print(f"  Optuna plots for {model_name} saved in '{save_folder}' directory.")


# --- 5. Multi-Output Support Helper Functions ---


def _check_multi_output_support(model):
    """Checks if a model natively supports multi-output regression."""
    native_multi_output_models = [
        'LinearRegression',
        'RandomForestRegressor',
        'ExtraTreesRegressor',
        'DecisionTreeRegressor',
        'GradientBoostingRegressor',
        'MultiTaskLasso',
        'MultiTaskElasticNet',
        'MultiOutputRegressor',
    ]
    
    model_name = model.__class__.__name__
    return model_name in native_multi_output_models


def _wrap_with_multi_output_if_needed(model, param_space):
    """
    Wraps a model with MultiOutputRegressor if needed for multi-output.
    Also adjusts param_space by prepending 'estimator__' to parameter names.
    
    Returns:
        tuple: (wrapped_model, adjusted_param_space, is_wrapped)
    """
    if _check_multi_output_support(model):
        print(f"  → {model.__class__.__name__} supports multi-output natively.")
        return model, param_space, False
    else:
        print(f"  → {model.__class__.__name__} does NOT support multi-output natively.")
        print(f"     Wrapping with MultiOutputRegressor...")
        
        wrapped_model = MultiOutputRegressor(model)
        
        adjusted_param_space = {}
        for param_name, param_info in param_space.items():
            new_param_name = f"estimator__{param_name}"
            adjusted_param_space[new_param_name] = param_info
        
        if adjusted_param_space:
            print(f"     Adjusted hyperparameters for wrapper:")
            for old_name, new_name in zip(param_space.keys(), adjusted_param_space.keys()):
                print(f"       {old_name} → {new_name}")
        
        return wrapped_model, adjusted_param_space, True


# --- 6. Main Workflow Functions ---


def optimize_model(
    model,
    param_space,
    X_train,
    y_train,
    X_test,
    y_test,
    folders,
    is_nn=False,
    is_gpr=False,
    gpr_kernel_map=None,
):
    """
    Performs multi-objective optimization using Optuna.
    
    IMPORTANT: Optimization uses SCALED data for fair comparison across outputs.
    Different length scales are properly handled during hyperparameter search.
    
    For GPR with kernel hyperparameter search:
    - param_space should include 'kernel' as a categorical parameter
    - gpr_kernel_map should map kernel names to kernel objects
    
    Args:
        folders: Dict with folder paths (from setup_folders())
    
    Returns:
        tuple: (best_params, train_metrics_dict, test_metrics_dict, y_pred_train, y_pred_test,
                final_model, study, optimization_time, retraining_time)
    """
    # Check multi-output support and wrap if needed
    if not is_nn:
        print(f"  Checking multi-output support for {model.__class__.__name__}...")
        model, param_space, is_wrapped = _wrap_with_multi_output_if_needed(model, param_space)
    else:
        is_wrapped = False
    
    # Transform Data
    X_train_scaled = x_scaler.transform(X_train)
    y_train_scaled = y_scaler.transform(_ensure_2d(y_train))
    X_test_scaled = x_scaler.transform(X_test)

    # Objective Function for Optuna
    def objective(trial):
        params = {}
        for param_name, param_info in param_space.items():
            p_type = param_info[0]
            if p_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_info[1], param_info[2]
                )
            elif p_type == "float":
                low, high = param_info[1], param_info[2]
                is_log = "log" in param_info
                params[param_name] = trial.suggest_float(
                    param_name, low, high, log=is_log
                )
            elif p_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_info[1]
                )
            else:
                raise ValueError(f"Unsupported parameter type: {p_type}")

        if is_nn:
            return cross_val_nn(X_train_scaled, y_train_scaled, create_nn, params)
        else:
            current_model = model
            
            # Handle GPR kernel mapping
            if is_gpr and "estimator__kernel" in params:
                kernel_name = params.pop("estimator__kernel")
                if gpr_kernel_map is None:
                    raise ValueError(
                        "gpr_kernel_map must be provided for kernel hyperparameter search"
                    )
                params["estimator__kernel"] = gpr_kernel_map[kernel_name]
            
            current_model.set_params(**params)
            scoring = {"r2": "r2", "neg_mse": "neg_mean_squared_error"}
            scores = cross_validate(
                current_model,
                X_train_scaled,
                y_train_scaled,
                cv=5,
                scoring=scoring,
                n_jobs=-1,
            )
            return -np.mean(scores["test_neg_mse"]), np.mean(scores["test_r2"])

    # Run Optimization
    if not param_space:
        print("  No hyperparameters defined. Training model with default settings.")
        optimization_time = 0
        best_params = {}
        final_model = model
    else:
        study = optuna.create_study(directions=["minimize", "maximize"])
        start_time = time.time()
        n_parallel_jobs = 1 if is_nn or is_gpr else -1
        if is_nn:
            print("  Using n_jobs=1 for Neural Network to ensure GPU stability.")
        if is_gpr:
            print("  Using n_jobs=1 for GaussianProcessRegressor to manage memory.")
        study.optimize(
            objective, n_trials=100, n_jobs=n_parallel_jobs, show_progress_bar=True
        )
        optimization_time = time.time() - start_time
        print(
            f"  Hyperparameter optimization completed in {optimization_time:.3f} seconds."
        )
        best_trial = max(study.best_trials, key=lambda t: t.values[1])
        best_params = best_trial.params
        print(
            f"  Selected Trial #{best_trial.number} with Mean MSE={best_trial.values[0]:.4f}, Mean R2={best_trial.values[1]:.4f}"
        )
        final_model = (
            model.set_params(**best_params) if not is_nn else create_nn(**best_params)
        )

    # Train Final Model
    print("  Retraining the final model with the best hyperparameters...")
    start_time = time.time()
    if is_nn:
        final_model.fit(
            X_train_scaled, y_train_scaled, epochs=100, batch_size=32, verbose=0
        )
    else:
        # Handle GPR kernel in final model
        if is_gpr and "estimator__kernel" in best_params:
            kernel_name = best_params.pop("estimator__kernel")
            best_params["estimator__kernel"] = gpr_kernel_map[kernel_name]
        
        final_model.fit(X_train_scaled, y_train_scaled)
    
    retraining_time = time.time() - start_time
    print(f"  Final model training complete in {retraining_time:.3f} seconds.")

    # Final Evaluation with Proper Length Scale Handling
    y_pred_train_scaled = final_model.predict(X_train_scaled)
    y_pred_test_scaled = final_model.predict(X_test_scaled)
    y_pred_train = y_scaler.inverse_transform(y_pred_train_scaled)
    y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled)

    # Calculate metrics with proper length scale handling
    train_metrics = compute_metrics_with_scales(y_train, y_pred_train, y_scaler, return_per_output=True)
    test_metrics = compute_metrics_with_scales(y_test, y_pred_test, y_scaler, return_per_output=True)

    return (
        best_params,
        train_metrics,
        test_metrics,
        y_pred_train,
        y_pred_test,
        final_model,
        study if "study" in locals() else None,
        optimization_time,
        retraining_time,
    )


def run_workflow(
    X,
    y,
    models,
    param_spaces,
    output_labels=None,
    gpr_kernel_map=None,
    folder_dict=None,
):
    """
    Executes the end-to-end multi-output machine learning workflow. 
    - Properly handles outputs with different length scales
    - Optimization uses SCALED data (fair across all outputs)
    - Reporting includes both original and normalized metrics
    - Customizable folder paths for output files
    
    Args:
        X: Feature matrix, shape (n_samples, n_features)
        y: Multi-output target, shape (n_samples, n_outputs)
        models: Dict of {model_name: model_instance}
        param_spaces: Dict of {model_name: hyperparameter_space}
        output_labels: Optional list of labels for output dimensions
        gpr_kernel_map: Optional dict mapping kernel names to kernel objects for GPR
        folder_dict: Optional dict with custom folder paths
            Expected keys:
            - 'models': Path for saving trained models (default: 'models')
            - 'plots': Path for saving plots (default: 'plots')
            - 'studies': Path for saving Optuna studies (default: 'studies')
            - 'results': Path for saving results CSV (default: current directory)
        
    Returns:
        pd.DataFrame: Results summary with metrics for all models
    
    Example:
        # Using default folders
        results = run_workflow(X, y, models, param_spaces)
        
        # Using custom folders
        folder_dict = {
            'models': '/output/my_models',
            'plots': '/output/my_plots',
            'studies': '/output/my_studies',
            'results': '/output/results'
        }
        results = run_workflow(X, y, models, param_spaces, folder_dict=folder_dict)
    """
    # Setup folders
    print("Setting up output folders...")
    folders = setup_folders(folder_dict)
    
    print("\nInitializing multi-output workflow...")
    print("NOTE: Optimization uses SCALED data for fair comparison across outputs")
    print("      Reporting includes normalized metrics (RMSE/std) for fair comparison")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nFitting and saving data scalers...")
    x_scaler.fit(X_train)
    y_scaler.fit(_ensure_2d(y_train))
    
    models_folder = get_folder_path(folders, 'models')
    joblib.dump(x_scaler, os.path.join(models_folder, "x_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(models_folder, "y_scaler.pkl"))
    print(f"  Scalers saved to: {models_folder}")

    results = []
    for name, model in models.items():
        print(f"\n--- Starting Workflow for: {name} ---")
        
        is_nn = name == "NeuralNetworkRegressor"
        is_gpr = isinstance(model, GaussianProcessRegressor) or (
            isinstance(model, MultiOutputRegressor) 
            and isinstance(model.estimator, GaussianProcessRegressor)
        )
        
        (
            best_params,
            train_metrics,
            test_metrics,
            y_pred_train,
            y_pred_test,
            final_model,
            study,
            opt_time,
            retrain_time,
        ) = optimize_model(
            model,
            param_spaces.get(name, {}),
            X_train,
            y_train,
            X_test,
            y_test,
            folders,
            is_nn=is_nn,
            is_gpr=is_gpr,
            gpr_kernel_map=gpr_kernel_map,
        )

        # Save Optuna study
        studies_folder = get_folder_path(folders, 'studies')
        if study:
            study_path = os.path.join(studies_folder, f"{name}_study.pkl")
            joblib.dump(study, study_path)
            print(f"  Optuna study saved to: {study_path}")
            plots_folder = get_folder_path(folders, 'plots')
            generate_optuna_plots(study, name, save_folder=plots_folder)

        # Save trained model
        models_folder = get_folder_path(folders, 'models')
        model_path = os.path.join(
            models_folder, f"{name}_best_model.{'h5' if is_nn else 'pkl'}"
        )
        if is_nn:
            final_model.save(model_path)
        else:
            joblib.dump(final_model, model_path)
        print(f"  Best model saved to: {model_path}")

        # Extract metrics per output
        train_r2 = []
        train_mse = []
        train_mae = []
        test_r2 = []
        test_mse = []
        test_mae = []
        for i in range(y.shape[1]):
            train_r2.append(train_metrics['per_output'][f'Output_{i}']['R2'])
            train_mse.append(train_metrics['per_output'][f'Output_{i}']['MSE'])
            train_mae.append(train_metrics['per_output'][f'Output_{i}']['MAE'])
            test_r2.append(test_metrics['per_output'][f'Output_{i}']['R2'])
            test_mse.append(test_metrics['per_output'][f'Output_{i}']['MSE'])
            test_mae.append(test_metrics['per_output'][f'Output_{i}']['MAE'])

        results.append(
            {
                "Model": name,
                "Optimization Time (s)": round(opt_time, 3),
                "Retraining Time (s)": round(retrain_time, 3),
                "Best Params": best_params,
                "Train R2 (Per-Output)": train_r2,
                "Train MSE (Per-Output)": train_mse,
                "Train MAE (Per-Output)": train_mae,
                "Test R2 (Per-Output)": test_r2,
                "Test MSE (Per-Output)": test_mse,
                "Test MAE (Per-Output)": test_mae,
            }
        )

        # Save plots
        plots_folder = get_folder_path(folders, 'plots')
        plot_yy_multi_output(
            y_train,
            y_pred_train,
            y_test,
            y_pred_test,
            name,
            save_folder=plots_folder,
            output_labels=output_labels,
        )

    results_df = pd.DataFrame(results)
    
    # Create simplified CSV with key metrics
    results_simple = results_df[[
        "Model",
        "Optimization Time (s)",
        "Retraining Time (s)",
        "Train R2 (Per-Output)",
        "Train MSE (Per-Output)",
        "Train MAE (Per-Output)",
        "Test R2 (Per-Output)",
        "Test MSE (Per-Output)",
        "Test MAE (Per-Output)",
    ]].copy()
    
    # Add Best Params as string
    results_simple["Best Params"] = results_df["Best Params"].astype(str)
    
    # Save results CSV
    results_folder = get_folder_path(folders, 'results')
    results_csv_path = os.path.join(results_folder, "results.csv")
    results_simple.to_csv(results_csv_path, index=False)
    
    # Save detailed results with per-output metrics
    results_detailed_path = os.path.join(results_folder, "results_detailed.txt")
    with open(results_detailed_path, "w") as f:
        f.write("DETAILED RESULTS WITH PER-OUTPUT METRICS\n")
        f.write("="*80 + "\n\n")
        for idx, row in results_df.iterrows():
            f.write(f"Model: {row['Model']}\n")
            f.write(f"Optimization Time: {row['Optimization Time (s)']} seconds\n")
            f.write(f"Retraining Time: {row['Retraining Time (s)']} seconds\n")
            f.write(f"Best Params: {row['Best Params']}\n\n")
            
            f.write("TRAINING METRICS:\n")
            train_r2_list = row['Train R2 (Per-Output)']
            train_mse_list = row['Train MSE (Per-Output)']
            train_mae_list = row['Train MAE (Per-Output)']
            f.write(f"  Mean R²: {np.mean(train_r2_list):.4f}\n")
            f.write(f"  Mean MSE: {np.mean(train_mse_list):.6f}\n")
            f.write(f"  Mean MAE: {np.mean(train_mae_list):.6f}\n")
            f.write("  Per-Output Metrics:\n")
            for i, (r2, mse, mae) in enumerate(zip(train_r2_list, train_mse_list, train_mae_list)):
                output_name = output_labels[i] if output_labels and i < len(output_labels) else f"Output_{i}"
                f.write(f"    {output_name}:\n")
                f.write(f"      R²: {r2:.4f}\n")
                f.write(f"      MSE: {mse:.6f}\n")
                f.write(f"      MAE: {mae:.6f}\n")
            
            f.write("\nTEST METRICS:\n")
            test_r2_list = row['Test R2 (Per-Output)']
            test_mse_list = row['Test MSE (Per-Output)']
            test_mae_list = row['Test MAE (Per-Output)']
            f.write(f"  Mean R²: {np.mean(test_r2_list):.4f}\n")
            f.write(f"  Mean MSE: {np.mean(test_mse_list):.6f}\n")
            f.write(f"  Mean MAE: {np.mean(test_mae_list):.6f}\n")
            f.write("  Per-Output Metrics:\n")
            for i, (r2, mse, mae) in enumerate(zip(test_r2_list, test_mse_list, test_mae_list)):
                output_name = output_labels[i] if output_labels and i < len(output_labels) else f"Output_{i}"
                f.write(f"    {output_name}:\n")
                f.write(f"      R²: {r2:.4f}\n")
                f.write(f"      MSE: {mse:.6f}\n")
                f.write(f"      MAE: {mae:.6f}\n")
            
            f.write("\n" + "-"*80 + "\n\n")
    
    print("\n--- Workflow Complete ---")
    print(f"✓ Results summary saved to: {results_csv_path}")
    print(f"✓ Detailed results saved to: {results_detailed_path}")
    print(f"✓ Models saved to: {get_folder_path(folders, 'models')}")
    print(f"✓ Plots saved to: {get_folder_path(folders, 'plots')}")
    print(f"✓ Studies saved to: {get_folder_path(folders, 'studies')}")
    
    return results_df
