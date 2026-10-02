# resource_allocation/qehvi_first_policy.py
"""
QEHVIFirstDecisionMaker — two-call LLM decision maker.

Philosophy
----------
qEHVI already balances exploration and exploitation within the regions it
believes the Pareto front is likely to expand into. qUCB's role is to explore
design-space regions the model hasn't seen, discovering Pareto candidates qEHVI
hasn't been directed toward.

The agent's primary job is stagnation/crowding detection, not assessing which
acquisition function is "effective" in general. The key diagnostic is HV/pt
(hypervolume gain per new Pareto point), trended over recent history.

Urgency is expressed as a continuous function of absolute iterations remaining —
no discrete phases, no cliff edges.

Pipeline
--------
Call 1 (before options computed): Assess HV/pt trend → choose beta
Call 2 (after options computed):  Choose allocation option (default: Option 0)
"""

from __future__ import annotations

import json
import os
import re
import time
import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from resource_allocation.decision_maker import DecisionMaker
from resource_allocation.decision_state import DecisionState, StrategyOutcome

logger = logging.getLogger(__name__)


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
        self._times.append(now)


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
# Baked-in framing blocks
# ---------------------------------------------------------------------------

_ACQ_FRAMING = """\
─── ACQUISITION FUNCTIONS ───────────────────────────────────────────────────
qEHVI already balances exploration and exploitation within the regions it
believes the Pareto front is likely to expand into. It naturally pushes toward
unexplored areas when those areas have high improvement potential. It is the
right default in most situations.

qUCB serves a different purpose: it explores regions of the design space that
have not been evaluated much, regardless of whether the GP currently predicts
them to be Pareto-improving. Its value is in discovering areas the model has
not seen yet — regions that may contain Pareto-optimal points qEHVI has never
been directed toward.

β (beta) controls how explorative qUCB is: larger β weights selection toward
points with greater GP uncertainty, pushing further into unmapped territory.
Default β = 2.0.

MutualInfo (shown alongside each option in the allocation table) measures how
much the batch reduces GP uncertainty about the optimum. The absolute value is
not interpretable — only compare it across options within the same iteration.
A higher MutualInfo option queries regions the GP understands least, regions
qEHVI tends to skip because it follows the surrogate's existing beliefs. It
does not directly guarantee better Pareto results, but it signals where the
GP's knowledge is thinnest and where qEHVI is most likely to be under-sampling.
─────────────────────────────────────────────────────────────────────────────"""

_URGENCY_FRAMING = """\
─── URGENCY GUIDANCE ────────────────────────────────────────────────────────
As fewer iterations remain, decisions should shift from probing uncertain
regions toward acting on what the GP already knows is good.
  • ~20+ iterations left : exploration carries low cost; testing uncertain
    regions is worthwhile
  • ~10 iterations left  : begin prioritizing high-confidence improvements;
    reduce speculative exploration
  • ~5 or fewer left     : exploit near-exclusively — too few evaluations
    remain to recover from a speculative bet that doesn't pay off

If an event reduced remaining iterations sharply, treat urgency as if the
campaign has always been at this point — the new iteration count is what
matters, not the original budget.
─────────────────────────────────────────────────────────────────────────────"""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _hv_per_pt(o: StrategyOutcome) -> Optional[float]:
    if o.pareto_points_added > 0:
        return o.actual_improvement / o.pareto_points_added
    return None


def _fmt_history(state: DecisionState) -> str:
    """
    Format history table with HV/pt and beta columns.
    Events (recent_events) are appended below the table rather than inline,
    since we don't have per-row event timestamps for historical rows.
    """
    outcomes = state.strategy_outcomes
    if not outcomes:
        return "  (no completed iterations yet)"

    lines = []
    for o in outcomes:
        hv_pct = (o.actual_improvement / o.hv_before * 100) if o.hv_before > 1e-9 else 0.0
        hpp = _hv_per_pt(o)
        hpp_str = f"+{hpp:.4f}" if hpp is not None and hpp >= 0 else (f"{hpp:.4f}" if hpp is not None else "—")
        pts_str = f"{o.pareto_points_added:+d}" if o.pareto_points_added != 0 else "0"
        lines.append(
            f"  Iter {o.iteration:>3}: {o.n_exploit} qEHVI + {o.n_explore} qUCB "
            f"→ {pts_str} Pareto pts, "
            f"HV {o.hv_before:.4f}→{o.hv_after:.4f} ({hv_pct:+.1f}%), "
            f"HV/pt={hpp_str}, β={o.beta_selected:.1f}"
        )

    result = "\n".join(lines)

    if state.recent_events:
        result += "\n\nEvents fired this iteration (affecting remaining resources):"
        for e in state.recent_events:
            result += f"\n  • {e}"

    if state.upcoming_events:
        result += "\n\nUpcoming events (next iteration):"
        for e in state.upcoming_events:
            result += f"\n  ⚠ {e}"

    result += (
        "\n\nHV/pt = HV change ÷ new Pareto points added. "
        "\"—\" means no new Pareto points that iteration (exclude from trend analysis)."
    )
    return result


def _fmt_options(state: DecisionState) -> str:
    lines = []
    for i, opt in enumerate(state.allocation_options.options):
        n_exp  = getattr(opt, "num_exploitation", "?")
        n_exr  = getattr(opt, "num_exploration",  "?")
        ehvi   = getattr(opt, "hypervolume_improvement", float("nan"))
        mi     = getattr(opt, "information_gain",        float("nan"))
        marker = "  ← DEFAULT" if i == 0 else ""
        lines.append(
            f"  Option {i}: {n_exp} qEHVI + {n_exr} qUCB "
            f"| qEHVI score={ehvi:.4f} | MutualInfo={mi:.4f}{marker}"
        )
    return "\n".join(lines)


def _iters_remaining(state: DecisionState) -> str:
    """Estimate iterations remaining from budget and batch size."""
    batch: Optional[int] = None

    # Prefer batch size from allocation options (available in Call 2)
    if state.allocation_options is not None:
        try:
            opt0 = state.allocation_options.options[0]
            b = getattr(opt0, "total_batch_size", None)
            if b:
                batch = int(b)
        except Exception:
            pass

    # Fall back to inferring batch size from history (available in Call 1)
    if batch is None and state.strategy_outcomes:
        last = state.strategy_outcomes[-1]
        b = last.n_exploit + last.n_explore
        if b > 0:
            batch = b

    if batch and state.cost_per_point > 0:
        try:
            cost_per_iter = float(batch) * state.cost_per_point
            budget_iters = int(state.budget_remaining / cost_per_iter)
            if state.time_per_point and state.time_per_point > 0:
                time_iters = int(state.time_remaining / state.time_per_point)
                return f"~{min(budget_iters, time_iters)}"
            return f"~{budget_iters}"
        except Exception:
            pass

    return "unknown"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QEHVIFirstDecisionMaker(DecisionMaker):
    """
    Two-call LLM decision maker that focuses on HV/pt trend as the stagnation
    signal and uses continuous urgency framing anchored to absolute iters remaining.

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
    ):
        self.llm = llm
        self.problem_description = problem_description
        self.objective_names = objective_names or ["Objective 1", "Objective 2"]
        self.log_dir = log_dir
        self.temperature = temperature
        self._rate_limiter = _RateLimiter(
            min_seconds=min_seconds_between_calls,
            max_qpm=max_calls_per_minute,
        )
        self._decision_history: List[Dict] = []

        # Per-iteration cache shared between select_beta and make_decision
        self._cached_iter: Optional[int] = None
        self._cached_assessment: str = ""
        self._cached_beta: float = 2.0

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_beta(self, state: DecisionState) -> float:
        """Call 1: assess HV/pt trend and select beta. Caches assessment for make_decision."""
        try:
            assessment, beta = self._call1_assessment_and_beta(state)
        except Exception as exc:
            logger.warning(
                "[QEHVIFirst] Call 1 error at iter %d: %s — using defaults",
                state.iteration, exc,
            )
            assessment = ""
            beta = 2.0

        self._cached_iter = state.iteration
        self._cached_assessment = assessment
        self._cached_beta = beta
        return beta

    def make_decision(self, state: DecisionState) -> Tuple[int, str, Optional[Dict]]:
        num_options = len(state.allocation_options.options)

        if self._cached_iter == state.iteration and self._cached_assessment:
            assessment = self._cached_assessment
        else:
            # select_beta wasn't called first — run Call 1 now
            try:
                assessment, beta = self._call1_assessment_and_beta(state)
                self._cached_iter = state.iteration
                self._cached_assessment = assessment
                self._cached_beta = beta
            except Exception as exc:
                logger.warning(
                    "[QEHVIFirst] Call 1 (late) error at iter %d: %s", state.iteration, exc
                )
                assessment = ""

        try:
            option_idx, reasoning = self._call2_option_selection(state, assessment)
        except Exception as exc:
            logger.warning(
                "[QEHVIFirst] Call 2 error at iter %d: %s — falling back to Option 0",
                state.iteration, exc,
            )
            option_idx = 0
            reasoning = f"[Fallback due to error: {exc}]"

        option_idx = max(0, min(option_idx, num_options - 1))

        record = {
            "iteration": state.iteration,
            "call1_assessment": assessment,
            "call1_beta": self._cached_beta,
            "call2_selected_option": option_idx,
            "call2_reasoning": reasoning,
            "budget_remaining": getattr(state, "budget_remaining", None),
            "time_remaining": getattr(state, "time_remaining", None),
        }
        self._decision_history.append(record)
        self._log(state.iteration, record)

        return option_idx, reasoning, None

    def cleanup(self):
        if self.log_dir and self._decision_history:
            path = os.path.join(self.log_dir, "decision_history.json")
            with open(path, "w") as f:
                json.dump(self._decision_history, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Call 1 — Assessment + Beta
    # ------------------------------------------------------------------

    def _call1_assessment_and_beta(self, state: DecisionState) -> Tuple[str, float]:
        history_text = _fmt_history(state)
        iters_left = _iters_remaining(state)

        objs = self.objective_names
        objectives_str = (
            objs[0] if len(objs) == 1
            else ", ".join(objs[:-1]) + f" and {objs[-1]}"
        )

        ps = state.pareto_status
        pareto_summary = (
            f"{ps.num_points} points, HV={ps.hypervolume:.4f}"
            if ps else "(not yet available)"
        )

        prompt = f"""You are guiding a Bayesian Optimization campaign.
Problem: {self.problem_description}
Objectives: {objectives_str}

{_ACQ_FRAMING}

─── ITERATION HISTORY (last {len(state.strategy_outcomes)} completed) ──────
{history_text}
─────────────────────────────────────────────────────────────────────────────

─── RESOURCES ───────────────────────────────────────────────────────────────
  Iterations completed : {state.iteration}
  Estimated iters left : {iters_left}
  Budget remaining     : ${state.budget_remaining:.2f}  (${state.cost_per_point:.2f}/point)
  Time remaining       : {state.time_remaining:.1f} weeks
  Current Pareto front : {pareto_summary}
─────────────────────────────────────────────────────────────────────────────

{_URGENCY_FRAMING}

─── YOUR TASK ───────────────────────────────────────────────────────────────
Examine the HV/pt column in the history above, but read it carefully in context.

STEP 1 — Identify which iterations were deliberate exploration.
Rows where β > 2.0 or qUCB points > 0 represent intentional exploration
investments. Low or zero HV/pt on those rows is EXPECTED and NORMAL — qUCB was
not targeting the Pareto front, it was probing uncertain regions. Do not count
these rows as evidence of crowding or stagnation.

STEP 2 — Assess the trend on qEHVI-primary iterations only.
Focus on rows where β ≈ 2.0 and qUCB = 0 (or very few). Among those rows:

  Crowding signal: HV/pt declining while Pareto points are still being added.
  qEHVI is refining regions it already knows rather than finding new territory.
  Adding qUCB points (with higher β) can probe unmapped design-space regions
  that may contain better Pareto candidates.

  Stagnation signal: zero new Pareto points across multiple consecutive
  qEHVI-primary iterations. qEHVI is stuck entirely. qUCB with a higher β is
  needed to break the pattern.

  Healthy signal: HV/pt stable or rising. qEHVI is finding new territory.
  Keep β = 2.0.

STEP 3 — Check whether past exploration paid off (one-iteration lag).
After an exploration iteration, the GP model is retrained on the new data.
qEHVI at the NEXT iteration then exploits what was just learned. This means the
payoff from exploration appears in the row AFTER the high-β row, not the
high-β row itself.

If the most recent iteration used high β or qUCB:
  → Do NOT raise β further yet. Give qEHVI one iteration to exploit the updated
    model. Default to β = 2.0 unless the row AFTER the exploration also shows
    continued stagnation or crowding in qEHVI-primary terms.

If a past exploration was followed by a recovery in HV/pt → it worked. If the
row after the exploration still showed flat HV/pt → the uncertain region was not
hiding better Pareto candidates, and further exploration in the same vein is
unlikely to help.

STEP 4 — Factor in iterations remaining.
With very few iterations left, raising β is riskier — less time to recover if
the exploration does not pay off. When in doubt with few iters left, prefer
β = 2.0.
─────────────────────────────────────────────────────────────────────────────

Respond using EXACTLY this format:

ASSESSMENT: <1-2 sentences describing what the HV/pt trend shows — crowding, stagnation, or healthy expansion>
BETA: <number 1–100; default is 2.0>
REASONING: <1 sentence explaining the beta choice>"""

        self._rate_limiter.wait()
        print(f"\n{'─'*60}")
        print(f"[QEHVIFirst Call 1 — Assessment + Beta] Iteration {state.iteration}")
        print(f"{'─'*60}")
        print(prompt)
        response = _call_llm(self.llm, prompt, self.temperature)
        print(f"\n[Call 1 — Response]\n{response}")
        print(f"{'─'*60}")

        assessment = self._parse_assessment(response)
        beta = self._parse_beta(response)
        return assessment, beta

    # ------------------------------------------------------------------
    # Call 2 — Option Selection
    # ------------------------------------------------------------------

    def _call2_option_selection(self, state: DecisionState, assessment: str) -> Tuple[int, str]:
        num_options = len(state.allocation_options.options)
        options_text = _fmt_options(state)
        iters_left = _iters_remaining(state)

        prompt = f"""You are selecting a resource allocation for a Bayesian Optimization campaign.

─── SITUATION ASSESSMENT ────────────────────────────────────────────────────
{assessment if assessment else "(no prior assessment available — treat as healthy expansion)"}
─────────────────────────────────────────────────────────────────────────────

─── AVAILABLE OPTIONS ───────────────────────────────────────────────────────
{options_text}

Note: qEHVI score and MutualInfo are on different scales — compare each metric
across options separately; do not compare them to each other. The MutualInfo
absolute value is not meaningful on its own: only use it to rank options against
each other. A notably higher MutualInfo on explore-heavy options signals those
options query regions the GP knows least — use this alongside the crowding/
stagnation assessment when deciding how many qUCB points to include.
─────────────────────────────────────────────────────────────────────────────

─── DECISION RULES ──────────────────────────────────────────────────────────
Option 0 (all qEHVI) is the DEFAULT. Choose it unless the assessment above
identifies crowding or stagnation that warrants qUCB exploration.

  • Crowding detected → choose an option with some qUCB points; scale the
    number of qUCB points to the severity of the signal
  • Stagnation detected → use a larger qUCB share
  • Estimated iters left is very low (~5 or fewer) → prefer Option 0 regardless
    of crowding; qEHVI is more reliable when evaluations are precious
  • No clear signal, or early in the campaign → Option 0
─────────────────────────────────────────────────────────────────────────────

Resources: est. iters left = {iters_left} | budget ${state.budget_remaining:.2f}

Respond using EXACTLY this format:

SELECTED_OPTION: <integer 0 to {num_options - 1}>
REASONING: <1-2 sentences connecting the assessment to this specific choice>"""

        self._rate_limiter.wait()
        print(f"\n{'─'*60}")
        print(f"[QEHVIFirst Call 2 — Option Selection] Iteration {state.iteration}")
        print(f"{'─'*60}")
        print(prompt)
        response = _call_llm(self.llm, prompt, self.temperature)
        print(f"\n[Call 2 — Response]\n{response}")
        print(f"{'─'*60}")

        return self._parse_decision(response, num_options)

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_assessment(self, text: str) -> str:
        m = re.search(
            r"ASSESSMENT:\s*(.+?)(?=\nBETA:|\nREASONING:|$)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if m:
            return m.group(1).strip()
        return text.split("\n")[0].strip()

    def _parse_beta(self, text: str) -> float:
        m = re.search(r"BETA:\s*([\d.]+)", text, re.IGNORECASE)
        if m:
            try:
                return max(1.0, min(100.0, float(m.group(1))))
            except ValueError:
                pass
        logger.warning("[QEHVIFirst] Could not parse BETA from Call 1, using 2.0")
        return 2.0

    def _parse_decision(self, text: str, num_options: int) -> Tuple[int, str]:
        m = re.search(r"SELECTED_OPTION:\s*(\d+)", text, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            if not (0 <= idx < num_options):
                logger.warning(
                    "[QEHVIFirst] Option %d out of range [0,%d), falling back to 0",
                    idx, num_options,
                )
                idx = 0
        else:
            logger.warning("[QEHVIFirst] Could not parse SELECTED_OPTION, falling back to 0")
            idx = 0

        m = re.search(r"REASONING:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
        reasoning = m.group(1).strip() if m else text.strip()
        return idx, reasoning

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, iteration: int, record: Dict):
        if not self.log_dir:
            return
        path = os.path.join(self.log_dir, f"qehvi_first_iteration_{iteration}.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)
