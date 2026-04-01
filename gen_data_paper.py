"""
export_to_excel.py

Scans a root folder for experiment directories, computes per-seed / per-iteration
statistics for every metric, and writes everything to a single .xlsx file with
one sheet per experiment. No plotting is done — all raw values are exported so
you can chart them manually in Excel.

Usage
-----
1. Set ROOT_DIR to the folder that contains your experiment directories.
2. Set OUTPUT_FILE to wherever you want the .xlsx saved.
3. Run:  python export_to_excel.py

Sheet layout (one sheet per experiment directory)
--------------------------------------------------
Each metric gets its own block of rows separated by a blank row.

Block header row:   Metric: <metric_name>
Column header row:  Iteration | Seed_1 | Seed_2 | ... | Mean | Std | CI_Lower | CI_Upper

Then one row per iteration containing:
  - raw value from every individual seed (NaN if that seed was missing for that iteration)
  - cross-seed mean, std, and 95 % CI bounds (computed with a t-distribution)

Metrics exported
----------------
  best_score          – best single-point hypervolume indicator
  total_hypervolume   – full Pareto-front hypervolume (if present)
  combined_uncertainty – obj1 + obj2 total uncertainty (if present)
  selected_n_opt      – exploitation candidate count (agent strategies only)
  selected_n_exp      – exploration candidate count (agent strategies only)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import openpyxl
import pandas as pd
import scipy.stats as stats
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# ===========================================================================
# CONFIGURATION — edit these two lines
# ===========================================================================

ROOT_DIR = Path("test")          # folder that contains experiment dirs
OUTPUT_FILE = Path("experiment_test.xlsx")  # output .xlsx path

# ===========================================================================
# STRATEGIES / LABELS
# (edit if your folder names differ)
# ===========================================================================

STRATEGIES = ["Agent_MultiStage", "qEHVI", "qUCB"]

LABELS = {
    "Agent_MultiStage": "3-Stage Agent",
    "qEHVI": "qEHVI",
    "qUCB":  "qUCB",
}

CONFIDENCE_LEVEL = 0.95

# ===========================================================================
# DATA LOADING
# ===========================================================================

def find_seed_dirs(exp_dir: Path) -> List[Path]:
    return sorted(d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("seed"))


def load_convergence_data(exp_dir: Path) -> Dict[str, List[pd.DataFrame]]:
    data: Dict[str, List[pd.DataFrame]] = {s: [] for s in STRATEGIES}
    for seed_dir in find_seed_dirs(exp_dir):
        for strategy in STRATEGIES:
            csv_path = seed_dir / strategy / f"{strategy}_convergence.csv"
            if csv_path.exists():
                data[strategy].append(pd.read_csv(csv_path))
    return data


# ===========================================================================
# STATISTICS HELPERS
# ===========================================================================

def _ci(values: List[float], confidence: float = CONFIDENCE_LEVEL):
    n = len(values)
    if n < 2:
        return 0.0
    t = stats.t.ppf((1 + confidence) / 2, n - 1)
    return t * np.std(values, ddof=1) / np.sqrt(n)


def per_iter_raw(dfs: List[pd.DataFrame], col: str) -> Dict[int, List[Optional[float]]]:
    """Return {iteration: [val_seed0, val_seed1, ...]} preserving seed order (None if missing)."""
    if not dfs:
        return {}
    valid = [df for df in dfs if col in df.columns]
    if not valid:
        return {}
    max_iter = int(max(df["iteration"].max() for df in valid))
    result: Dict[int, List[Optional[float]]] = {}
    for it in range(max_iter + 1):
        row_vals: List[Optional[float]] = []
        for df in valid:
            match = df[df["iteration"] == it]
            row_vals.append(float(match[col].values[0]) if len(match) else None)
        result[it] = row_vals
    return result


def per_iter_combined_uncertainty(dfs: List[pd.DataFrame]) -> Dict[int, List[Optional[float]]]:
    cols = {"total_uncertainty_obj1", "total_uncertainty_obj2"}
    valid = [df for df in dfs if cols.issubset(df.columns)]
    if not valid:
        return {}
    max_iter = int(max(df["iteration"].max() for df in valid))
    result: Dict[int, List[Optional[float]]] = {}
    for it in range(max_iter + 1):
        row_vals: List[Optional[float]] = []
        for df in valid:
            match = df[df["iteration"] == it]
            if len(match):
                r = match.iloc[0]
                row_vals.append(float(r["total_uncertainty_obj1"]) + float(r["total_uncertainty_obj2"]))
            else:
                row_vals.append(None)
        result[it] = row_vals
    return result


# ===========================================================================
# BELIEF DATA
# ===========================================================================

EFFECTIVENESS_MAP = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}


def load_belief_data(exp_dir: Path, seeds: List[str], agent: str) -> Dict[int, Dict[str, List[float]]]:
    iter_data: Dict = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        log_dir = exp_dir / seed / agent / "agent_logs"
        if not log_dir.exists():
            continue
        for file in sorted(log_dir.glob("iteration_*.json")):
            try:
                iteration = int(file.stem.split("_")[-1])
                with open(file) as f:
                    beliefs = json.load(f).get("beliefs", {})
                iter_data[iteration]["exp_eff"].append(EFFECTIVENESS_MAP[beliefs["exploration_effectiveness"]])
                iter_data[iteration]["exp_conf"].append(float(beliefs["exploration_confidence"]))
                iter_data[iteration]["expt_eff"].append(EFFECTIVENESS_MAP[beliefs["exploitation_effectiveness"]])
                iter_data[iteration]["expt_conf"].append(float(beliefs["exploitation_confidence"]))
            except (KeyError, ValueError):
                continue
    return iter_data


# ===========================================================================
# XLSX WRITING HELPERS
# ===========================================================================

HEADER_FILL   = PatternFill("solid", start_color="1F4E79", end_color="1F4E79")
METRIC_FILL   = PatternFill("solid", start_color="2E75B6", end_color="2E75B6")
ALT_ROW_FILL  = PatternFill("solid", start_color="D6E4F0", end_color="D6E4F0")
STAT_FILL     = PatternFill("solid", start_color="E2EFDA", end_color="E2EFDA")

WHITE_BOLD    = Font(name="Arial", bold=True, color="FFFFFF", size=10)
BOLD          = Font(name="Arial", bold=True, size=10)
NORMAL        = Font(name="Arial", size=10)
CENTER        = Alignment(horizontal="center", vertical="center")
LEFT          = Alignment(horizontal="left",   vertical="center")


def _write_cell(ws, row, col, value, font=None, fill=None, alignment=None):
    cell = ws.cell(row=row, column=col, value=value)
    if font:      cell.font      = font
    if fill:      cell.fill      = fill
    if alignment: cell.alignment = alignment
    return cell


def write_metric_block(
    ws,
    start_row: int,
    metric_label: str,
    seed_labels: List[str],
    raw_data: Dict[int, List[Optional[float]]],  # {iteration: [per-seed values]}
) -> int:
    """
    Write one metric block starting at start_row.
    Returns the next available row after the block + a blank separator.
    """
    if not raw_data:
        return start_row

    iterations = sorted(raw_data.keys())
    n_seeds = max(len(v) for v in raw_data.values())

    # --- metric header row ---
    cell = ws.cell(row=start_row, column=1, value=f"Metric: {metric_label}")
    cell.font  = WHITE_BOLD
    cell.fill  = METRIC_FILL
    cell.alignment = LEFT
    # merge across all data columns
    total_cols = 1 + n_seeds + 4   # Iteration + seeds + Mean/Std/CI_Lo/CI_Hi
    ws.merge_cells(start_row=start_row, start_column=1,
                   end_row=start_row,   end_column=total_cols)
    start_row += 1

    # --- column header row ---
    col_headers = ["Iteration"] + seed_labels[:n_seeds] + ["Mean", "Std Dev", "CI Lower (95%)", "CI Upper (95%)"]
    for c, h in enumerate(col_headers, 1):
        _write_cell(ws, start_row, c, h, font=WHITE_BOLD, fill=HEADER_FILL, alignment=CENTER)
    start_row += 1

    # --- data rows ---
    for idx, it in enumerate(iterations):
        vals = raw_data[it]
        fill = ALT_ROW_FILL if idx % 2 == 0 else None
        _write_cell(ws, start_row, 1, it, font=BOLD, fill=fill, alignment=CENTER)

        clean = [v for v in vals if v is not None]
        for c, v in enumerate(vals, 2):
            _write_cell(ws, start_row, c, v if v is not None else "", font=NORMAL, fill=fill, alignment=CENTER)

        # summary stats
        stat_col = 2 + n_seeds
        mean_val = float(np.mean(clean)) if clean else None
        std_val  = float(np.std(clean, ddof=1)) if len(clean) > 1 else (0.0 if clean else None)
        ci       = _ci(clean) if len(clean) > 1 else 0.0
        ci_lo    = (mean_val - ci) if mean_val is not None else None
        ci_hi    = (mean_val + ci) if mean_val is not None else None

        for offset, v in enumerate([mean_val, std_val, ci_lo, ci_hi]):
            _write_cell(ws, start_row, stat_col + offset, v, font=NORMAL, fill=STAT_FILL, alignment=CENTER)

        start_row += 1

    return start_row + 1  # blank separator


def write_belief_block(ws, start_row: int, seed_labels: List[str],
                       belief_data: Dict[int, Dict[str, List[float]]]) -> int:
    """Write belief dynamics (exp/expt effectiveness + confidence) from agent logs."""
    if not belief_data:
        return start_row

    keys = [
        ("exp_eff",  "Exploration Effectiveness (0=Low, 1=Med, 2=High)"),
        ("exp_conf", "Exploration Confidence"),
        ("expt_eff", "Exploitation Effectiveness (0=Low, 1=Med, 2=High)"),
        ("expt_conf","Exploitation Confidence"),
    ]

    # Collect max seeds across all iterations for sizing
    max_seeds = max(
        (len(belief_data[it][k]) for it in belief_data for k, _ in keys if belief_data[it][k]),
        default=0
    )
    used_seeds = min(max_seeds, len(seed_labels))

    for key, label in keys:
        raw: Dict[int, List[Optional[float]]] = {}
        for it in sorted(belief_data.keys()):
            vals = belief_data[it].get(key, [])
            raw[it] = [float(v) for v in vals] + [None] * (used_seeds - len(vals))
        start_row = write_metric_block(ws, start_row, f"Agent Belief – {label}",
                                       seed_labels[:used_seeds], raw)
    return start_row


# ===========================================================================
# PER-EXPERIMENT SHEET WRITER
# ===========================================================================

def write_experiment_sheet(wb: openpyxl.Workbook, exp_dir: Path):
    sheet_name = exp_dir.name[:31]  # Excel sheet name limit
    ws = wb.create_sheet(title=sheet_name)

    print(f"  Processing: {exp_dir.name}")

    data = load_convergence_data(exp_dir)
    seed_dirs = find_seed_dirs(exp_dir)
    seed_labels = [d.name for d in seed_dirs]

    # Sheet title
    ws.cell(row=1, column=1, value=f"Experiment: {exp_dir.name}").font = Font(name="Arial", bold=True, size=13)
    ws.cell(row=2, column=1, value=f"Seeds: {len(seed_dirs)}   |   Strategies: {', '.join(LABELS[s] for s in STRATEGIES if s in LABELS)}").font = NORMAL
    current_row = 4

    # ------------------------------------------------------------------ #
    # One section per strategy
    # ------------------------------------------------------------------ #
    for strategy in STRATEGIES:
        dfs = data.get(strategy, [])
        if not dfs:
            continue

        label = LABELS.get(strategy, strategy)

        # Strategy section header
        cell = ws.cell(row=current_row, column=1, value=f"Strategy: {label}")
        cell.font = Font(name="Arial", bold=True, size=12, color="FFFFFF")
        cell.fill = PatternFill("solid", start_color="375623", end_color="375623")
        cell.alignment = LEFT
        ws.merge_cells(start_row=current_row, start_column=1,
                       end_row=current_row, end_column=20)
        current_row += 1

        # Best score
        raw = per_iter_raw(dfs, "best_score")
        current_row = write_metric_block(ws, current_row,
                                         "Best Single-Point Hypervolume (best_score)",
                                         seed_labels, raw)

        # Total hypervolume (optional)
        raw_hv = per_iter_raw(dfs, "total_hypervolume")
        if raw_hv:
            current_row = write_metric_block(ws, current_row,
                                              "Pareto Front Hypervolume (total_hypervolume)",
                                              seed_labels, raw_hv)

        # Combined uncertainty (optional)
        raw_unc = per_iter_combined_uncertainty(dfs)
        if raw_unc:
            current_row = write_metric_block(ws, current_row,
                                              "Combined Uncertainty (obj1 + obj2)",
                                              seed_labels, raw_unc)

        # Selection counts (agent strategies only)
        if "agent" in strategy.lower():
            for col, label_suffix in [("selected_n_opt", "Exploitation Candidates (n_opt)"),
                                       ("selected_n_exp", "Exploration Candidates (n_exp)")]:
                raw_sel = per_iter_raw(dfs, col)
                if raw_sel:
                    current_row = write_metric_block(ws, current_row,
                                                      label_suffix, seed_labels, raw_sel)

            # Belief dynamics from agent logs
            belief_data = load_belief_data(exp_dir, seed_labels, strategy)
            if belief_data:
                current_row = write_belief_block(ws, current_row, seed_labels, belief_data)

        current_row += 1  # extra spacer between strategies

    # Auto-size columns (heuristic)
    for col_cells in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col_cells[0].column)
        for cell in col_cells:
            try:
                max_len = max(max_len, len(str(cell.value or "")))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 30)

    # Freeze the top rows and first column for easier navigation
    ws.freeze_panes = "B4"


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    root = ROOT_DIR.resolve()
    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {root}")

    # Collect experiment directories (any subdir that contains at least one seed* folder)
    exp_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and any(
            sub.is_dir() and sub.name.startswith("seed") for sub in d.iterdir()
        )
    )

    if not exp_dirs:
        raise ValueError(f"No experiment directories (with seed* subdirs) found in {root}")

    print(f"Found {len(exp_dirs)} experiment(s) in {root}")

    wb = openpyxl.Workbook()
    # Remove the default empty sheet
    wb.remove(wb.active)

    for exp_dir in exp_dirs:
        write_experiment_sheet(wb, exp_dir)

    out = OUTPUT_FILE.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"\nSaved: {out}")
    print(f"Sheets: {wb.sheetnames}")


if __name__ == "__main__":
    main()