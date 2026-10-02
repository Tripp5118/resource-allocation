# resource_allocation/adaptive_scout_policy.py
"""
AdaptiveScout v2 — free LLM decision maker for Bayesian Optimization.

Philosophy
----------
The LLM's value is making judgment calls in ambiguous situations, not executing
pre-specified rules. This policy provides:
  - Raw computed signals (no pre-digested labels)
  - Rolling iteration history (sequence visibility, ~6 iterations)
  - Pareto composition coverage (domain reasoning via element ranges)
  - Principled guidance text (not enforced constraints)

The only Python enforcement is output parsing and option bounds checking.
Early-campaign guidance (prefer low beta / qEHVI when GP is poorly fitted)
is delivered as prompt text grounded in BO theory — not as a code lockout.

Two LLM calls per iteration (architecturally required)
-------------------------------------------------------
Call 1 (select_beta): Beta selection.
  LLM sees campaign state, optimization signals, Pareto composition coverage,
  rolling history, and beta guidance. Outputs BETA + REASONING.
  Options do not yet exist — they are computed by Python using this beta.

Call 2 (make_decision): Option selection.
  LLM sees the beta it chose, its own reasoning from Call 1, current resources,
  and the options table with actual qEHVI + MI scores at the chosen beta.
  Outputs SELECTED_OPTION + REASONING.
"""

from __future__ import annotations

import json
import os
import re
import time
import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from resource_allocation.decision_maker import DecisionMaker
from resource_allocation.decision_state import DecisionState, StrategyOutcome

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guidance text constants
# ---------------------------------------------------------------------------

_ACQ_FRAMING = """\
qEHVI focuses on what the GP currently believes is near-Pareto-optimal. It is the
reliable Pareto front improver and the right default in most situations.

qUCB with elevated β scouts design-space regions the GP has not characterized well —
regions with high posterior uncertainty. Its purpose is to find productive territory
qEHVI has not yet been directed toward. Exploration will often produce no short-term HV
gain. That is acceptable when it locates new productive regions faster than qEHVI
alone would. It is NOT acceptable when exploration is saturated (MI declining) or
when the campaign is nearly complete."""

_BETA_GUIDANCE = """\
β controls how far into uncertain regions qUCB will probe, measured in posterior
standard deviations. GP inputs and outputs are normalized, so β has the same
meaning regardless of problem scale or physical units.

  β ≈ 2:   Conservative. qUCB candidates stay near regions the GP already knows well.
            Right default when HV is actively improving.
  β = 3–5: Moderate exploration. Noticeably more uncertainty-seeking.
            Appropriate when HV progress has slowed but MI signals remaining territory.
  β = 5–8: Aggressive scouting. Sends candidates into poorly-characterized regions.
            Justified when HV has materially stalled and MI is still elevated.
  β > 8:   Only in extreme stagnation with a strong MI signal.

Early in the campaign (first ~20-25% of resources consumed), the GP posterior is
poorly constrained — uncertainty estimates are unreliable and elevated beta may send
candidates to arbitrary regions rather than genuinely informative ones. Prefer β
near 2 unless you observe a specific signal that justifies deviation.

β only has effect if the selected option also allocates exploration points.
Choosing high β with 0 explore points in the batch is a no-op."""


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    def __init__(self, min_seconds: float = 0.0, max_qpm: Optional[int] = None):
        self.min_gap = min_seconds
        self.qpm = max_qpm
        self._last: float = 0.0
        self._times: Deque[float] = deque()

    def wait(self):
        now = time.monotonic()
        gap = self.min_gap - (now - self._last)
        if gap > 0:
            time.sleep(gap)
            now = time.monotonic()
        if self.qpm is not None:
            cutoff = now - 60.0
            while self._times and self._times[0] < cutoff:
                self._times.popleft()
            if len(self._times) >= self.qpm:
                sleep_for = self._times[0] + 60.0 - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
            now = time.monotonic()
        self._last = now
        if self.qpm is not None:
            self._times.append(now)


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

def _call_llm(llm, prompt: str, temperature: float) -> str:
    try:
        if hasattr(llm, "invoke"):
            if temperature is not None and hasattr(llm, "temperature"):
                orig = llm.temperature
                llm.temperature = temperature
                result = llm.invoke(prompt)
                llm.temperature = orig
            else:
                result = llm.invoke(prompt)
            return result.content if hasattr(result, "content") else str(result)
        return str(llm(prompt))
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Formatting helpers (module-level)
# ---------------------------------------------------------------------------

def _fmt_options_table(state: DecisionState) -> str:
    if state.allocation_options is None:
        return "  (options not yet computed)"
    lines = []
    for i, opt in enumerate(state.allocation_options.options):
        n_exp = getattr(opt, "num_exploitation", "?")
        n_exr = getattr(opt, "num_exploration",  "?")
        ehvi  = getattr(opt, "hypervolume_improvement", float("nan"))
        mi    = getattr(opt, "information_gain",        float("nan"))
        lines.append(
            f"  Option {i}: {n_exp} qEHVI + {n_exr} qUCB"
            f" | qEHVI={ehvi:.4f} | MI={mi:.4f}"
        )
    return "\n".join(lines)


def _compute_iters_remaining(state: DecisionState) -> int:
    batch = None
    if state.allocation_options is not None:
        try:
            b = getattr(state.allocation_options.options[0], "total_batch_size", None)
            if b:
                batch = int(b)
        except Exception:
            pass
    if batch is None and state.strategy_outcomes:
        last = state.strategy_outcomes[-1]
        b = last.n_exploit + last.n_explore
        if b > 0:
            batch = b
    if batch and state.cost_per_point > 0:
        try:
            cost_iter = float(batch) * state.cost_per_point
            budget_iters = int(state.budget_remaining / cost_iter)
            if state.time_per_point and state.time_per_point > 0:
                time_iters = int(state.time_remaining / state.time_per_point)
                return min(budget_iters, time_iters)
            return budget_iters
        except Exception:
            pass
    return 0


def _parse_beta(text: str) -> Optional[float]:
    m = re.search(r"BETA:\s*([\d.]+)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _parse_reasoning(text: str) -> str:
    m = re.search(r"REASONING:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _parse_decision(text: str, num_options: int) -> Tuple[int, str]:
    m = re.search(r"SELECTED_OPTION:\s*(\d+)", text, re.IGNORECASE)
    if m:
        idx = int(m.group(1))
        if not (0 <= idx < num_options):
            logger.warning("[AdaptiveScout] Option %d out of range, fallback 0", idx)
            idx = 0
    else:
        logger.warning("[AdaptiveScout] Could not parse SELECTED_OPTION, fallback 0")
        idx = 0
    m = re.search(r"REASONING:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    reasoning = m.group(1).strip() if m else text.strip()
    return idx, reasoning


def _print_call(label: str, iteration: int, prompt: str):
    print(f"\n{'─'*60}\n{label} Iteration {iteration}\n{'─'*60}")
    print(prompt)


def _print_response(response: str):
    print(f"\n[Response]\n{response}\n{'─'*60}")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AdaptiveScoutDecisionMaker(DecisionMaker):
    """
    Two-call LLM decision maker. No Python-enforced state machine or hard constraints.
    The LLM receives raw signals, rolling history, and Pareto composition coverage,
    and makes unconstrained beta and option decisions guided by prompt text.

    Parameters
    ----------
    llm :
        LangChain BaseChatModel or plain callable (prompt: str) -> str.
    problem_description : str
    objective_names : list[str]
    log_dir : str | None
    min_seconds_between_calls : float
    max_calls_per_minute : int | None
    temperature : float
    major_elements : list[str] | None
        Elements that are primary contributors (e.g. ["Fe", "Co", "Ni"]).
        If provided, mean ± std across the Pareto front is shown for these.
    all_elements : list[str] | None
        All elements in composition order (e.g. ["Fe","Co","Ni","V","Mo","C","Nb","Ti","Si"]).
        If provided, per-element min/max ranges across the Pareto front are shown.
        If None, the Pareto composition block is omitted entirely.
    """

    def __init__(
        self,
        llm,
        problem_description: str = "",
        objective_names: Optional[List[str]] = None,
        log_dir: Optional[str] = None,
        min_seconds_between_calls: float = 1.0,
        max_calls_per_minute: Optional[int] = None,
        temperature: float = 0.3,
        major_elements: Optional[List[str]] = None,
        all_elements: Optional[List[str]] = None,
    ):
        self.llm = llm
        self.problem_description = problem_description
        self.objective_names = objective_names or ["Objective 1", "Objective 2"]
        self.log_dir = log_dir
        self.temperature = temperature
        self.major_elements = major_elements
        self.all_elements = all_elements

        self._rate_limiter = _RateLimiter(
            min_seconds=min_seconds_between_calls,
            max_qpm=max_calls_per_minute,
        )

        # Signal history
        self._hv_values: List[float] = []
        self._mi_values: List[float] = []
        self._mi_at_start: Optional[float] = None

        # Per-iteration cache (shared between select_beta and make_decision)
        self._cached_iter: Optional[int] = None
        self._cached_beta: float = 2.0
        self._cached_signals: Dict = {}
        self._call1_reasoning: str = ""

        self._decision_history: List[Dict] = []
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_beta(self, state: DecisionState) -> float:
        if state.pareto_status:
            self._hv_values.append(state.pareto_status.hypervolume)

        signals = self._compute_signals(state)
        self._cached_iter = state.iteration
        self._cached_signals = signals

        try:
            beta = self._call1(state, signals)
        except Exception as exc:
            logger.warning(
                "[AdaptiveScout] Call 1 error at iter %d: %s — fallback beta=2.0",
                state.iteration, exc,
            )
            beta = 2.0
            self._call1_reasoning = f"[Fallback: {exc}]"

        self._cached_beta = beta
        return beta

    def make_decision(
        self, state: DecisionState
    ) -> Tuple[int, str, Optional[Dict]]:
        if self._cached_iter != state.iteration:
            self.select_beta(state)

        beta    = self._cached_beta
        signals = self._cached_signals

        self._record_mi(state)

        try:
            option_idx, reasoning = self._call2(state, beta, signals)
        except Exception as exc:
            logger.warning(
                "[AdaptiveScout] Call 2 error at iter %d: %s — fallback 0",
                state.iteration, exc,
            )
            option_idx = 0
            reasoning = f"[Fallback: {exc}]"

        num_options = len(state.allocation_options.options)
        option_idx = max(0, min(option_idx, num_options - 1))

        record = {
            "iteration":    state.iteration,
            "beta":         beta,
            "option":       option_idx,
            "reasoning_c1": self._call1_reasoning,
            "reasoning_c2": reasoning,
            "signals":      {
                k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in signals.items()
                if v is not None
            },
        }
        self._decision_history.append(record)
        self._log(state.iteration, record)

        return option_idx, reasoning, None

    def cleanup(self):
        if self.log_dir and self._decision_history:
            path = os.path.join(self.log_dir, "adaptive_scout_history.json")
            with open(path, "w") as f:
                json.dump(self._decision_history, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_signals(self, state: DecisionState) -> Dict:
        sig: Dict = {}

        # ── HV velocity ──────────────────────────────────────────────
        delta_hvs = [
            self._hv_values[i] - self._hv_values[i - 1]
            for i in range(1, len(self._hv_values))
        ]

        if len(delta_hvs) >= 3:
            recent_rate = float(np.mean(delta_hvs[-3:]))
            all_windows = [
                float(np.mean(delta_hvs[i: i + 3]))
                for i in range(len(delta_hvs) - 2)
            ]
            baseline = float(np.median(all_windows)) if all_windows else recent_rate
            vel_ratio = recent_rate / baseline if abs(baseline) > 1e-10 else 1.0
            sig["velocity_ratio"]   = round(vel_ratio, 3)
            sig["recent_hv_rate"]   = round(recent_rate, 6)
            sig["baseline_hv_rate"] = round(baseline, 6)
        else:
            sig["velocity_ratio"]   = None
            sig["recent_hv_rate"]   = None
            sig["baseline_hv_rate"] = None

        # ── MI signals ───────────────────────────────────────────────
        if self._mi_values:
            mi_curr  = self._mi_values[-1]
            mi_start = self._mi_at_start if (self._mi_at_start and
                                              self._mi_at_start > 1e-10) else mi_curr
            sig["mi_current"]  = round(mi_curr, 6)
            sig["mi_vs_start"] = round(mi_curr / max(mi_start, 1e-10), 3)
        else:
            sig["mi_current"]  = None
            sig["mi_vs_start"] = None

        # ── Resource signals ─────────────────────────────────────────
        init_b = getattr(state, "initial_budget", None)
        init_t = getattr(state, "initial_time",   None)
        fracs  = []
        if init_b and init_b > 0:
            fracs.append(state.budget_remaining / init_b)
        if init_t and init_t > 0:
            fracs.append(state.time_remaining / init_t)
        rf = min(fracs) if fracs else None
        sig["resource_fraction_remaining"] = round(rf, 3) if rf is not None else None
        sig["iters_remaining"] = _compute_iters_remaining(state)

        if init_t and init_t > 0 and state.time_remaining is not None:
            sig["time_pct_remaining"] = round(state.time_remaining / init_t * 100, 1)
        else:
            sig["time_pct_remaining"] = None

        return sig

    def _record_mi(self, state: DecisionState):
        if state.allocation_options is None:
            return
        opts = state.allocation_options.options
        if not opts:
            return
        mi = getattr(opts[-1], "information_gain", 0.0)
        if self._mi_at_start is None:
            self._mi_at_start = mi
        self._mi_values.append(mi)

    # ------------------------------------------------------------------
    # History and coverage formatting
    # ------------------------------------------------------------------

    def _fmt_recent_history(self, state: DecisionState, n: int = 6) -> str:
        outcomes = state.strategy_outcomes[-n:] if state.strategy_outcomes else []
        if not outcomes:
            return "  (no completed iterations yet — this is early in the campaign)"

        # Align MI values: _mi_values[i] corresponds to iteration i+1
        # Take the last len(outcomes) values
        mi_aligned = self._mi_values[-len(outcomes):] if self._mi_values else []
        # Pad with None if fewer MI values than outcomes
        mi_padded = [None] * (len(outcomes) - len(mi_aligned)) + list(mi_aligned)

        lines = []
        for o, mi in zip(outcomes, mi_padded):
            hv_delta = o.hv_after - o.hv_before
            pts = f"{o.pareto_points_added:+d}" if o.pareto_points_added != 0 else " 0"
            mi_str = f"{mi:.4f}" if mi is not None else "  N/A"
            lines.append(
                f"  Iter {o.iteration:>3}: {o.n_exploit} qEHVI + {o.n_explore} qUCB"
                f" | β={o.beta_selected:.1f}"
                f" | HV: {o.hv_before:.4f}→{o.hv_after:.4f} (Δ{hv_delta:+.4f})"
                f" | MI: {mi_str}"
                f" | {pts} Pareto pts"
            )
        return "\n".join(lines)

    def _fmt_pareto_coverage(self, pareto_X: np.ndarray) -> str:
        if self.all_elements is None or pareto_X is None or len(pareto_X) == 0:
            return ""

        n = len(pareto_X)
        all_els = self.all_elements
        n_cols = min(len(all_els), pareto_X.shape[1])

        lines = [f"=== PARETO FRONT COMPOSITION COVERAGE (N={n} points) ==="]

        # Major contributors: mean ± std
        if self.major_elements:
            major_stats = []
            for el in self.major_elements:
                if el in all_els:
                    col = all_els.index(el)
                    if col < n_cols:
                        vals = pareto_X[:, col] * 100.0
                        if np.max(vals) < 0.1:
                            continue
                        major_stats.append(f"  {el}: {np.mean(vals):.1f} ± {np.std(vals):.1f}%")
            if major_stats:
                lines.append("Major contributors (mean ± std across Pareto front):")
                lines.append("\n".join(major_stats))
                lines.append("")

        # All elements: min-max ranges, 4 per row — skip elements always at 0
        range_parts = []
        for col, el in enumerate(all_els):
            if col >= n_cols:
                break
            vals = pareto_X[:, col] * 100.0
            if np.max(vals) < 0.1:  # skip fixed-at-zero and inactive elements
                continue
            range_parts.append(f"{el}: {np.min(vals):.0f}–{np.max(vals):.0f}%")

        lines.append("Element ranges across Pareto front:")
        row, chunk = [], 4
        for i, part in enumerate(range_parts):
            row.append(f"  {part:<16}")
            if len(row) == chunk or i == len(range_parts) - 1:
                lines.append("".join(row))
                row = []

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM Call 1 — Beta selection
    # ------------------------------------------------------------------

    def _call1(self, state: DecisionState, signals: Dict) -> float:
        objectives_str  = ", ".join(self.objective_names)
        iters_remaining = signals.get("iters_remaining", "?")
        rf              = signals.get("resource_fraction_remaining")
        budget_pct      = f"{(1.0 - rf) * 100:.0f}" if rf is not None else "?"
        time_pct        = signals.get("time_pct_remaining")
        time_pct_str    = f"{time_pct:.0f}%" if time_pct is not None else "N/A"
        hv              = state.pareto_status
        hv_line         = (
            f"{hv.hypervolume:.4f} (Δ={hv.hypervolume_change:+.4f})"
            if hv else "(unavailable)"
        )
        pareto_size     = hv.num_points if hv else "?"
        pareto_change   = (
            f"{hv.points_added_last_iter:+d} points this iteration"
            if hv else "?"
        )

        vel_ratio = signals.get("velocity_ratio")
        vel_str   = f"{vel_ratio:.3f}" if vel_ratio is not None else "(insufficient history)"

        mi_curr      = signals.get("mi_current")
        mi_vs_start  = signals.get("mi_vs_start")
        mi_line      = f"{mi_curr:.4f}" if mi_curr is not None else "(no history yet)"
        mi_vs_str    = (
            f"{mi_vs_start:.1%} of iteration-1 value"
            if mi_vs_start is not None else "(insufficient history)"
        )

        recent_history = self._fmt_recent_history(state)

        pareto_coverage = ""
        if state.pareto_compositions is not None and self.all_elements:
            pareto_coverage = self._fmt_pareto_coverage(state.pareto_compositions)
            if pareto_coverage:
                pareto_coverage = "\n" + pareto_coverage + "\n"

        prompt = f"""You are selecting a beta parameter for a Bayesian Optimization campaign.

Problem: {self.problem_description}
Objectives: {objectives_str}

{_ACQ_FRAMING}

{_BETA_GUIDANCE}

=== CAMPAIGN STATE ===
Iteration          : {state.iteration}
Resources consumed : {budget_pct}%  |  Time remaining: {time_pct_str}  |  Iters remaining: ~{iters_remaining}
Pareto front HV    : {hv_line}
Pareto size        : {pareto_size} points ({pareto_change})

=== OPTIMIZATION SIGNALS ===
HV velocity ratio  : {vel_str}
  (recent mean ΔHV ÷ historical baseline — 1.0 = on pace, <0.5 = materially slowed, >1.0 = accelerating)
MI of best explore option: {mi_line}  ({mi_vs_str})
  (declining MI = design space increasingly well-characterized; elevated MI = unexplored territory remains)
{pareto_coverage}
=== RECENT HISTORY ===
{recent_history}

=== OUTPUT FORMAT ===
Output EXACTLY these two lines (nothing else):

BETA: <number>
REASONING: <one sentence citing specific signals above>"""

        self._rate_limiter.wait()
        _print_call("[AdaptiveScout Call 1]", state.iteration, prompt)
        response = _call_llm(self.llm, prompt, self.temperature)
        _print_response(response)

        beta = _parse_beta(response)
        if beta is None:
            logger.warning("[AdaptiveScout] Could not parse BETA at iter %d, fallback 2.0", state.iteration)
            beta = 2.0

        self._call1_reasoning = _parse_reasoning(response)
        return beta

    # ------------------------------------------------------------------
    # LLM Call 2 — Option selection
    # ------------------------------------------------------------------

    def _call2(
        self,
        state: DecisionState,
        beta: float,
        signals: Dict,
    ) -> Tuple[int, str]:
        num_options     = len(state.allocation_options.options)
        options_table   = _fmt_options_table(state)
        iters_remaining = _compute_iters_remaining(state) or signals.get("iters_remaining", "?")
        hv              = state.pareto_status
        hv_pts_str      = str(hv.num_points) if hv else "?"
        hv_hv_str       = f"{hv.hypervolume:.4f}" if hv else "?"
        rf              = signals.get("resource_fraction_remaining")
        budget_pct      = f"{(1.0 - rf) * 100:.0f}" if rf is not None else "?"
        time_pct        = signals.get("time_pct_remaining")
        time_pct_str    = f"{time_pct:.0f}%" if time_pct is not None else "N/A"

        prompt = f"""You are selecting a resource allocation option for a Bayesian Optimization campaign.

=== CONTEXT ===
Beta selected : {beta:.1f}
Your reasoning: {self._call1_reasoning}
Resources consumed: {budget_pct}%  |  Time remaining: {time_pct_str}
                    Iters remaining: ~{iters_remaining}
Pareto front  : {hv_pts_str} pts, HV={hv_hv_str}

=== HOW TO READ THE OPTIONS TABLE ===
Each option allocates the batch between qEHVI (exploitation) and qUCB (exploration) points.
qEHVI score: expected direct Pareto HV improvement — compare qEHVI values across options only.
MI score: expected information gain about unexplored regions — compare MI values across options only.
Do NOT compare qEHVI against MI — they are on different scales.
Higher-index options send more points to qUCB at β={beta:.1f}.
Options with more qUCB points trade qEHVI improvement for MI gain.

{options_table}

=== OUTPUT FORMAT ===
Output EXACTLY these two lines (nothing else):

SELECTED_OPTION: <integer 0 to {num_options - 1}>
REASONING: <1-2 sentences on why this option's qEHVI/MI tradeoff fits your beta reasoning>"""

        self._rate_limiter.wait()
        _print_call("[AdaptiveScout Call 2]", state.iteration, prompt)
        response = _call_llm(self.llm, prompt, self.temperature)
        _print_response(response)

        return _parse_decision(response, num_options)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, iteration: int, record: Dict):
        if not self.log_dir:
            return
        path = os.path.join(
            self.log_dir, f"adaptive_scout_iteration_{iteration}.json"
        )
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)
